import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import traceback
import difflib
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image
from frappe.utils import add_days, get_url_to_form, nowdate
import time


class InvoiceUpload(Document):
    def on_submit(self):
        try:
            self.reload()
            # Create draft invoice on submit
            self.create_invoice_from_child(submit_invoice=False)
            # Make document non-editable
            frappe.db.set_value("Invoice Upload", self.name, "docstatus", 1)
        except Exception as e:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            error_message = f"Invoice Creation Failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "Invoice Creation Failed")
            frappe.throw(f"Invoice creation failed: {str(e)}")

    def before_save(self):
        try:
            # Make submitted documents read-only
            if self.docstatus == 1:
                self.flags.read_only = True
                
            # AUTO-MATCH REPRESENTATIVE
            if self.representative_name and not self.representative:
                matched_user = self.fuzzy_match_representative(self.representative_name)
                if matched_user:
                    self.representative = matched_user
        except Exception as e:
            frappe.log_error(f"before_save failed: {str(e)}\n{traceback.format_exc()}", "Invoice Upload Error")

    def extract_invoice(self):
        try:
            if not self.file:
                frappe.throw("No file attached.")

            start_time = time.time()
            file_path = get_file_path(self.file)
            if not file_path:
                frappe.throw(f"File path not found for: {self.file}")
                
            text = ""

            # Enhanced Odoo-style preprocessing
            def preprocess_image(pil_img):
                try:
                    img = np.array(pil_img.convert("RGB"))
                    channels = img.shape[-1] if img.ndim == 3 else 1
                    
                    if channels == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                        
                    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(scaled)
                    thresh = cv2.adaptiveThreshold(
                        enhanced, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 15, 10
                    )
                    kernel = np.ones((3, 3), np.uint8)
                    processed = cv2.erode(thresh, kernel, iterations=1)
                    return processed
                except Exception as e:
                    frappe.log_error(f"Image processing failed: {str(e)}", "OCR Error")
                    return pil_img

            # Get OCR config from site config or use default
            ocr_config = frappe.conf.get("ocr_config", "--psm 4 --oem 3 -l eng+urd")

            if file_path.endswith(".pdf"):
                # Convert only first 3 pages to prevent timeouts
                images = convert_from_path(file_path, dpi=200, first_page=1, last_page=3)
                for img in images:
                    if time.time() - start_time > 120:  # 2-minute timeout
                        frappe.throw("OCR processing timed out. Try with smaller file or fewer pages.")
                    
                    processed = preprocess_image(img)
                    page_text = pytesseract.image_to_string(processed, config=ocr_config)
                    text += page_text
                    img.close()  # Free memory immediately
            else:
                img = Image.open(file_path)
                try:
                    processed = preprocess_image(img)
                    text = pytesseract.image_to_string(processed, config=ocr_config)
                finally:
                    img.close()
                    
            # Save extracted text for debugging
            self.raw_ocr_text = text[:10000]  # Save first 10k characters
            
            # ===== EXTRACT METADATA =====
            # Extract dates
            dates = self.extract_dates(text)
            if dates:
                self.invoice_date = dates.get("invoice_date") or self.invoice_date
                self.due_date = dates.get("due_date") or self.due_date
                self.delivery_date = dates.get("delivery_date") or self.delivery_date
            
            # Extract source and reference
            self.source = self.extract_source(text) or self.source
            self.reference = self.extract_reference(text) or self.reference
            
            # Extract representative
            self.representative_name = self.extract_representative(text) or self.representative_name
            
            # Save immediately to capture metadata
            self.save()
            # ===== END METADATA EXTRACTION =====
            
            items = self.extract_items(text)
            extracted_data = {
                "items": items,
                "party": None
            }

            # Get all items for matching
            all_items = self.get_items_for_matching()
            
            self.set("invoice_upload_item", [])
            seen_descriptions = set()  # Track seen descriptions to avoid duplicates
            
            for row in items:
                # Skip empty or invalid descriptions
                if not row["description"] or len(row["description"]) < 3:
                    continue
                    
                # Normalize description for duplicate check
                normalized_desc = re.sub(r'\W+', '', row["description"].lower())
                
                # Skip duplicate items
                if normalized_desc in seen_descriptions:
                    continue
                seen_descriptions.add(normalized_desc)
                
                # First try to match text in square brackets (item codes)
                bracket_match = None
                bracket_text = self.extract_bracket_text(row["description"])
                
                # Try matching with bracket text first
                if bracket_text:
                    bracket_match = self.fuzzy_match_item(bracket_text, all_items)
                    if bracket_match and bracket_match["score"] > 85:
                        matched_item = bracket_match["item_name"]
                        self.append("invoice_upload_item", {
                            "ocr_description": row["description"],
                            "qty": row["qty"],
                            "rate": row["rate"],
                            "ocr_uom": row.get("uom"),
                            "item": matched_item
                        })
                        # Match UOM for this row
                        child_rows = self.get("invoice_upload_item")
                        last_row = child_rows[-1]
                        if last_row.ocr_uom:
                            matched_uom = self.fuzzy_match_uom(last_row.ocr_uom)
                            last_row.uom = matched_uom
                        continue
                
                # If bracket match not found, try full description
                full_match = self.fuzzy_match_item(row["description"], all_items)
                if full_match and full_match["score"] > 75:
                    matched_item = full_match["item_name"]
                else:
                    matched_item = None
                    
                self.append("invoice_upload_item", {
                    "ocr_description": row["description"],
                    "qty": row["qty"],
                    "rate": row["rate"],
                    "ocr_uom": row.get("uom"),
                    "item": matched_item
                })
                
                # Match UOM for this row
                child_rows = self.get("invoice_upload_item")
                last_row = child_rows[-1]
                if last_row.ocr_uom:
                    matched_uom = self.fuzzy_match_uom(last_row.ocr_uom)
                    last_row.uom = matched_uom

            # Extract party with fuzzy matching
            party_name = self.extract_party(text)
            if party_name:
                party_match = self.fuzzy_match_party(party_name)
                if party_match:
                    extracted_data["party"] = party_match["name"]
                else:
                    extracted_data["party"] = party_name

            self.extracted_data = json.dumps(extracted_data, indent=2)
            self.ocr_status = "Extracted"
            self.save()
            frappe.msgprint("OCR Extraction completed. Please review data before submitting.")
            
            return {
                "status": "success",
                "items": items,
                "party": extracted_data["party"]
            }
        except Exception as e:
            error_message = f"Extraction failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "OCR Extraction Failed")
            frappe.throw(f"Extraction failed: {str(e)}")

    def ensure_party_exists(self):
        extracted = json.loads(self.extracted_data or '{}')
        party = extracted.get("party")

        if not party or not party.strip():
            frappe.throw("Party is missing. Cannot create invoice.")
        
        # Check if party exists
        if frappe.db.exists(self.party_type, party):
            self.party = party
            return
            
        # Try fuzzy matching again in case of close matches
        party_match = self.fuzzy_match_party(party)
        if party_match:
            self.party = party_match["name"]
            return

        # If no match found, throw error
        frappe.throw(f"Party '{party}' not found in the system. Please create it first.")

    def create_invoice_from_child(self, submit_invoice=False):
        """Create invoice, optionally submit it based on parameter"""
        # Check if invoice already created
        if self.invoice_created:
            frappe.throw("Invoice already created for this document")
            
        # Ensure party is set and exists
        # self.ensure_party_exists()

        # Create the appropriate invoice type
        if self.party_type == "Supplier":
            inv = frappe.new_doc("Purchase Invoice")
            inv.supplier = self.party
        else:
            inv = frappe.new_doc("Sales Invoice")
            inv.customer = self.party
            
        # SET REPRESENTATIVE
        if self.representative:
            inv.contact_person = self.representative

        # ===== START: CHANGED SOURCE/REFERENCE MAPPING =====
        # Handle Sales Invoice mapping
        if self.party_type != "Supplier":  # Customer/Sales Invoice
            inv.po_no = self.source or ""  # Source → Customer's Purchase Order
            inv.remarks = self.reference or ""  # Reference → Remarks
        else:  # Supplier/Purchase Invoice
            inv.remarks = self.source or ""  # Source → Remarks
            inv.bill_no = self.reference or self.name  # Reference → Bill No
            inv.bill_date = self.invoice_date  # Invoice Date → Bill Date
        # ===== END: CHANGED SOURCE/REFERENCE MAPPING =====

        # Get appropriate account based on invoice type
        if self.party_type == "Supplier":
            account = self.get_expense_account()
            account_field = "expense_account"
        else:
            account = self.get_income_account()
            account_field = "income_account"

        # Add items from the child table
        items_added = 0
        for row in self.invoice_upload_item:
            item_code = row.item
            if not item_code:
                frappe.msgprint(f"Skipping item: {row.ocr_description} - no item matched", alert=True)
                continue

            try:
                # Get item details
                item_doc = frappe.get_doc("Item", item_code)
                
                # PRIORITIZE MATCHED UOM IF AVAILABLE
                if row.uom:
                    uom = row.uom
                elif self.party_type == "Supplier":
                    uom = item_doc.purchase_uom or item_doc.stock_uom or "Nos"
                else:
                    uom = item_doc.sales_uom or item_doc.stock_uom or "Nos"
                
                # Create item dictionary
                item_dict = {
                    "item_code": item_code,
                    "item_name": item_doc.item_name,
                    "description": item_doc.description or row.ocr_description,
                    "qty": row.qty,
                    "rate": row.rate,
                    "uom": uom
                }
                
                # Set account field based on invoice type
                item_dict[account_field] = account
                
                inv.append("items", item_dict)
                items_added += 1
            except Exception as e:
                frappe.msgprint(f"Error adding item {item_code}: {str(e)}", alert=True, indicator="red")

        if items_added == 0:
            frappe.throw("No valid items found to create invoice")

        # SET DATES WITH OVERRIDE PRIORITY
        posting_date = self.posting_date or self.invoice_date or nowdate()
        due_date = self.due_date or add_days(posting_date, 30)
        
        # Set dates in invoice
        inv.posting_date = posting_date
        inv.due_date = due_date
        
        # For purchase invoices, set delivery date if available
        if self.party_type == "Supplier" and self.delivery_date:
            inv.set("delivery_date", self.delivery_date)
        
        # Calculate totals
        inv.run_method("set_missing_values")
        inv.run_method("calculate_taxes_and_totals")
        
        # Save invoice with appropriate validation
        try:
            # Bypass validations for draft invoices
            inv.flags.ignore_validate = True
            inv.flags.ignore_mandatory = True
            inv.insert(ignore_permissions=True)
            status = "Draft"
        except Exception as e:
            frappe.msgprint(f"Invoice creation failed: {str(e)}", alert=True, indicator="red")
            frappe.log_error(f"Invoice creation failed: {str(e)}", "Invoice Creation Error")
            return
        
        # Update status and reference
        frappe.db.set_value(self.doctype, self.name, {
            "invoice_created": 1,
            "invoice_reference": inv.name,
            "invoice_type": inv.doctype,
            "invoice_status": status
        })

        frappe.msgprint(f"<a href='{get_url_to_form(inv.doctype, inv.name)}'>{inv.name}</a> created ({status})")

    def get_expense_account(self):
        company = frappe.defaults.get_user_default("Company")
        account = frappe.db.get_value("Company", company, "default_expense_account")
        if not account:
            account = frappe.db.get_value("Account", {
                "account_type": "Expense",
                "company": company,
                "is_group": 0
            }, "name")
        if not account:
            frappe.throw("No default Expense Account found for the company.")
        return account

    def get_income_account(self):
        company = frappe.defaults.get_user_default("Company")
        account = frappe.db.get_value("Company", company, "default_income_account")
        if not account:
            account = frappe.db.get_value("Account", {
                "account_type": "Income",
                "company": company,
                "is_group": 0
            }, "name")
        if not account:
            frappe.throw("No default Income Account found for the company.")
        return account

    def extract_representative(self, text):
        """Extract representative name from invoice (FIXED)"""
        try:
            # First, try to find the representative using explicit patterns
            patterns = [
                r'Sales\s*Person\s*[:\-]?\s*([^\n\d]+)',
                r'Purchase\s*Representative\s*[:\-]?\s*([^\n\d]+)',
                r'Representative\s*[:\-]?\s*([^\n\d]+)',
                r'Prepared\s*by\s*[:\-]?\s*([^\n\d]+)',
                r'Attn\s*[:\-]?\s*([^\n\d]+)',
                r'ATTN\s*[:\-]?\s*([^\n\d]+)',
                r'Seller\s*[:\-]?\s*([^\n\d]+)',
                r'Purchaser\s*[:\-]?\s*([^\n\d]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Clean name and remove any trailing numbers/dates
                    name = re.sub(r'\s+\d+.*$', '', name)  # Remove trailing numbers
                    name = re.sub(r'[^\w\s\.-]', '', name).strip()
                    if name and len(name) > 3:
                        return name
            
            # If explicit patterns fail, look for name after "Purchase Representative"
            rep_match = re.search(r'Purchase Representative[^\n]*\n([^\n]+)', text, re.IGNORECASE)
            if rep_match:
                name = rep_match.group(1).strip()
                # Clean name and remove any trailing numbers/dates
                name = re.sub(r'\s+\d+.*$', '', name)  # Remove trailing numbers
                name = re.sub(r'[^\w\s\.-]', '', name).strip()
                if name and len(name) > 3:
                    return name
                    
            return None
        except Exception as e:
            frappe.log_error(f"extract_representative failed: {str(e)}\n{traceback.format_exc()}", "Representative Extraction Error")
            return None

    def fuzzy_match_representative(self, name):
        """Fuzzy match representative name to users"""
        if not name:
            return None
            
        clean_name = name.lower().strip()
        users = frappe.get_all("User", 
                              fields=["name", "full_name"],
                              filters={"enabled": 1})
        
        best_match = None
        best_score = 0
        
        for user in users:
            # Try both full name and username
            for field in ["full_name", "name"]:
                if not user.get(field):
                    continue
                    
                score = difflib.SequenceMatcher(
                    None, clean_name, user[field].lower()
                ).ratio() * 100
                
                if score > best_score:
                    best_score = score
                    best_match = user["name"]
        
        # Only return if good match
        return best_match if best_score > 75 else None

    def extract_items(self, text):
        # First try to extract as structured table items
        table_items = self.extract_table_items(text)
        if table_items:
            return table_items

        # Then try to extract as bill of charges
        charge_items = self.extract_charges(text)
        if charge_items:
            return charge_items

        # Fallback to context-aware extraction
        return self.extract_items_fallback(text)

    def extract_items_fallback(self, text):
        """Safe fallback that only looks in the item section with UOM extraction"""
        items = []
        
        # Find the start of items section
        start_index = text.find("DESCRIPTION")
        if start_index == -1:
            start_index = text.find("PARTICULARS")
        if start_index == -1:
            start_index = text.find("PRODUCT")
        if start_index == -1:
            start_index = 0
            
        # Find the end of items section
        end_index = text.find("Total", start_index)
        if end_index == -1:
            end_index = text.find("Subtotal", start_index)
        if end_index == -1:
            end_index = text.find("Payment", start_index)
        if end_index == -1:
            end_index = len(text)
            
        # Focus only on the item section
        item_section = text[start_index:end_index]
        
        # Look for quantity patterns in the item section with UOM
        # New regex pattern to capture quantity, UOM, and rate
        qty_matches = re.finditer(
            r'(\d+,\d+\.\d{3}|\d+\.\d{3}|\d+)\s*([a-zA-Z]{1,10})?\s+(\d+,\d+\.\d{2,3}|\d+\.\d{2,3}|\d+)',
            item_section, 
            re.IGNORECASE
        )
        
        for match in qty_matches:
            try:
                qty_str = match.group(1).replace(',', '')
                qty = float(qty_str)
                
                # Capture UOM if present
                uom = match.group(2) if match.group(2) else None
                
                # Get the full line
                line_start = item_section.rfind('\n', 0, match.start()) + 1
                line_end = item_section.find('\n', match.end())
                full_line = item_section[line_start:line_end].strip()
                
                # Extract description (everything before quantity)
                description = full_line.split(match.group(0))[0].strip()
                
                # Clean up description
                description = re.sub(r'^\W+|\W+$', '', description)
                description = re.sub(r'\s+', ' ', description)
                description = re.sub(r'\.{3,}', '', description)
                
                # Skip short descriptions
                if len(description) < 3:
                    continue
                
                # Extract rate from the match
                rate_str = match.group(3).replace(',', '')
                rate = float(rate_str)
                
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom  # Add UOM to item
                })
            except Exception as e:
                frappe.log_error(f"Item extraction failed: {str(e)}", "Item Extraction Error")
                continue
        
        return items

    def extract_table_items(self, text):
        """Extract items from structured tables with UOM support"""
        items = []
        lines = text.splitlines()
        
        # Find the start of the items table
        start_index = -1
        header_patterns = [
            r"DESCRIPTION.*QUANTITY.*UOM.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*UOM.*RATE.*AMOUNT",
            r"ITEM.*QTY.*UOM.*PRICE.*TOTAL",
            r"DESCRIPTION.*QUANTITY.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*RATE.*AMOUNT",
            r"ITEM.*QTY.*PRICE.*TOTAL"
        ]
        
        header_parts = None
        uom_col_index = None
        
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    start_index = i + 1
                    # Split header to find UOM column
                    if '|' in line:
                        header_parts = [part.strip() for part in line.split('|')]
                    else:
                        header_parts = re.split(r'\s{2,}', line)
                    
                    # Find UOM column index
                    for idx, part in enumerate(header_parts):
                        if "UOM" in part.upper():
                            uom_col_index = idx
                            break
                    break
            if start_index != -1:
                break
        
        # If table header found, process subsequent lines
        if start_index != -1:
            for i in range(start_index, min(start_index + 20, len(lines))):
                line = lines[i]
                if not line.strip() or "---" in line or "===" in line:
                    break
                
                # Split line by pipe character or multiple spaces
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')]
                else:
                    parts = re.split(r'\s{2,}', line)
                
                if len(parts) < 4:
                    continue
                
                try:
                    # Skip lines that are clearly not items
                    if "Total" in parts[0] or "Subtotal" in parts[0] or "Grand" in parts[0]:
                        continue
                    
                    # Combine description parts
                    description = " ".join(parts[:-3]).strip()
                    
                    # Last three parts should be Qty, Rate, Amount
                    qty_str = parts[-3].replace(',', '')
                    rate_str = parts[-2].replace(',', '')
                    
                    # Extract quantity number
                    qty_match = re.search(r'(\d+\.\d{3}|\d+)', qty_str)
                    if not qty_match:
                        continue
                    qty = float(qty_match.group(1))
                    
                    # Extract rate number
                    rate_match = re.search(r'(\d+\.\d{2,3}|\d+)', rate_str)
                    if not rate_match:
                        continue
                    rate = float(rate_match.group(1))
                    
                    # Clean up description
                    description = re.sub(r'\.{3,}', '', description)
                    description = re.sub(r'^\W+|\W+$', '', description)
                    
                    # Skip short descriptions
                    if len(description) < 3:
                        continue
                    
                    # Extract UOM if present
                    uom = None
                    if uom_col_index is not None and len(parts) > uom_col_index:
                        uom = parts[uom_col_index].strip()
                    
                    items.append({
                        "description": description,
                        "qty": qty,
                        "rate": rate,
                        "uom": uom  # Add UOM to item
                    })
                except Exception as e:
                    continue
        
        return items

    def extract_charges(self, text):
        """Extract items from bill of charges format"""
        items = []
        clean_text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Find the PARTICULARS section
        start = clean_text.find("PARTICULARS")
        if start == -1:
            return items

        # Extract table data
        table_pattern = r'Custom Duties(.+?)Service Charges'
        table_match = re.search(table_pattern, clean_text, re.DOTALL)
        if not table_match:
            return items
            
        table_text = table_match.group(1)
        
        # Process each charge line
        charge_pattern = r'(\w[\w\s\/-]+)\s+(\d{1,3}(?:,\d{3})*)\s+(\d{1,3}(?:,\d{3})*)'
        for match in re.finditer(charge_pattern, table_text):
            try:
                charge_name = match.group(1).strip()
                consignee_amount = float(match.group(2).replace(',', ''))
                balance_amount = float(match.group(3).replace(',', ''))
                total_amount = consignee_amount + balance_amount
                
                # Skip zero-amount lines
                if total_amount > 0:
                    items.append({
                        "description": charge_name,
                        "qty": 1,
                        "rate": total_amount,
                        "uom": None  # Charges typically don't have UOM
                    })
            except Exception:
                continue
                
        return items

    def fuzzy_match_uom(self, uom_text):
        """Fuzzy match UOM text to standard UOMs"""
        if not uom_text:
            return None
            
        clean_text = uom_text.lower().strip()
        uoms = frappe.get_all("UOM", fields=["name"])
        
        best_match = None
        best_score = 0
        
        for uom in uoms:
            score = difflib.SequenceMatcher(None, clean_text, uom.name.lower()).ratio() * 100
            if score > best_score:
                best_score = score
                best_match = uom.name
        
        # Only return if good match
        return best_match if best_score > 80 else None

    def extract_dates(self, text):
        """Extract dates from the invoice header with improved table parsing"""
        dates = {}
        # Try multi-line approach first
        lines = text.splitlines()
        header_found = False
        date_line = None
        
        # Look for the header line
        for i, line in enumerate(lines):
            if "Invoice Date" in line and "Due Date" in line and "Delivery Date" in line:
                header_found = True
                # The next line should contain the dates
                if i + 1 < len(lines):
                    date_line = lines[i + 1]
                break
        
        if header_found and date_line:
            # Extract dates from the value line
            date_values = re.findall(r'(\d{1,2}/\d{1,2}/\d{4})', date_line)
            if len(date_values) >= 3:
                try:
                    dates["invoice_date"] = self.convert_date_format(date_values[0])
                    dates["due_date"] = self.convert_date_format(date_values[1])
                    dates["delivery_date"] = self.convert_date_format(date_values[2])
                    return dates
                except Exception:
                    pass
        
        # Fallback to regex method for invoices without clear table structure
        header_match = re.search(
            r'Invoice\s+Date:.*?(\d{1,2}/\d{1,2}/\d{4}).*?'
            r'Due\s+Date:.*?(\d{1,2}/\d{1,2}/\d{4}).*?'
            r'Delivery\s+Date:.*?(\d{1,2}/\d{1,2}/\d{4})',
            text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if header_match:
            try:
                dates["invoice_date"] = self.convert_date_format(header_match.group(1))
                dates["due_date"] = self.convert_date_format(header_match.group(2))
                dates["delivery_date"] = self.convert_date_format(header_match.group(3))
            except Exception:
                frappe.log_error("Date extraction format error", "OCR Date Error")
        
        return dates

    def convert_date_format(self, date_str):
        """Convert DD/MM/YYYY to YYYY-MM-DD format"""
        try:
            day, month, year = date_str.split('/')
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except Exception:
            frappe.log_error(f"Invalid date format: {date_str}", "Date Conversion Error")
            return None

    def extract_source(self, text):
        """Extract source (PO/SO number) with enhanced pattern matching (FIXED)"""
        # 1. First look for Request for Quotation pattern
        rfq_match = re.search(r'Request\s*for\s*Quotation\s*[#:]*\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if rfq_match:
            return rfq_match.group(1).strip()
        
        # 2. Look for PO/SO patterns
        po_patterns = [
            r'\b(?:PO|SO)[\s-]*([A-Z0-9-]{8,})',  # Minimum 8 chars
            r'\b(?:Purchase|Sales)[\s-]*Order[\s-]*([A-Z0-9-]+)',
            r'Order[\s-]*Number[\s-]*:\s*([A-Z0-9-]+)',
            r'^[\s]*([A-Z]{2,}\d{4,}[-]?\d*)\s*$'  # Standalone PO pattern
        ]
        
        for pattern in po_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                po_value = match.group(1).strip()
                if len(po_value) > 5:  # Minimum length for PO
                    return po_value
        
        # 3. Table-based extraction as fallback
        lines = text.splitlines()
        order_keywords = [
            "Source", "PO", "SO", "Order", "Book",
            "Purchase Order", "Sales Order", "P.O.", "S.O.", "Request for Quotation"
        ]
        
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in order_keywords):
                if i + 1 < len(lines):
                    value_line = lines[i + 1]
                    
                    if '|' in value_line:
                        parts = [p.strip() for p in value_line.split('|') if p.strip()]
                    else:
                        parts = re.split(r'\s{2,}', value_line)
                    
                    if parts:
                        # Find the most PO-like value
                        for part in parts:
                            if re.match(r'[A-Z]{2,}[-]?[A-Z0-9-]{6,}', part):
                                return part.strip()
        
        return None

    def extract_reference(self, text):
        """Extract reference number from the invoice"""
        # 1. Look for invoice number pattern
        inv_patterns = [
            r'Invoice\s+([A-Z]+/\d{4}/\d{5})',           # Invoice INV/2025/00789
            r'Ref(?:erence)?\s*[:#]?\s*([A-Z0-9-]+)',     # Ref: ABC-123
            r'Document\s+Number\s*:\s*([A-Z0-9-]+)',      # Document Number: DOC-456
            r'Bill\s+No\.?\s*:\s*([A-Z0-9-]+)',           # Bill No: BILL-789
            r'^\s*([A-Z]{2,}\d{4,})\s*$'                  # Standalone reference like INV202500789
        ]
        
        for pattern in inv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 2. Look for payment communication reference
        payment_match = re.search(r'Payment\s+Communication:\s*([^\n]+)', text, re.IGNORECASE)
        if payment_match:
            return payment_match.group(1).strip()
            
        return None

    def extract_party(self, text):
        """Extract the actual partner name from the invoice"""
        # 1. First look for explicit "Partner Name" field
        partner_match = re.search(r'Partner\s*Name\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if partner_match:
            party = partner_match.group(1).strip()
            # Remove any trailing non-alphanumeric characters
            party = re.sub(r'[^\w\s\-]$', '', party).strip()
            if party:
                return party
        
        # 2. Look for name in square brackets
        bracket_match = re.search(r'\[([^\]]+)\]', text)
        if bracket_match:
            candidate = bracket_match.group(1).strip()
            # Check if it looks like a name (contains letters and spaces)
            if re.search(r'[a-zA-Z]', candidate) and ' ' in candidate:
                return candidate

        # 3. Look for the most prominent name in the top section
        # This is usually the customer/supplier name
        top_section = text.split("Invoice Date:")[0] if "Invoice Date:" in text else text[:500]
        
        # Find the longest word sequence that looks like a name
        name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', top_section)
        if name_candidates:
            # Get the longest candidate as it's likely the partner name
            name_candidates.sort(key=len, reverse=True)
            return name_candidates[0]

        # 4. Look for other common labels
        party_labels = ["Customer", "Client", "Supplier", "Vendor", "Bill To", "Sold To"]
        for label in party_labels:
            pattern = re.compile(fr'{label}\s*:\s*([^\n]+)', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                party = match.group(1).strip()
                party = re.sub(r'[^\w\s\-]$', '', party).strip()
                if party:
                    return party

        # 5. Look for a name-like string near the invoice title
        title_match = re.search(r'Invoice\s+\w+/\d+/\d+', text, re.IGNORECASE)
        if title_match:
            # Look before and after the title for a name
            start_pos = max(0, title_match.start() - 100)
            end_pos = min(len(text), title_match.end() + 100)
            context = text[start_pos:end_pos]
            
            # Find the most prominent name in this context
            name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
            if name_candidates:
                name_candidates.sort(key=len, reverse=True)
                return name_candidates[0]

        return None

    def get_items_for_matching(self):
        """Get all items with their names and codes for matching"""
        # Get all active items
        items = frappe.get_all("Item", 
                              fields=["item_code", "item_name"],
                              filters={"disabled": 0})
        
        # Create a list of all possible names and codes
        item_data = []
        for item in items:
            # Add item code as primary identifier
            if item.item_code:
                item_data.append({
                    "item_name": item.item_code,  # Actual item code
                    "match_text": re.sub(r'[\[\]]', '', item.item_code.lower()),
                    "type": "code"
                })
            
            # Add item name as secondary identifier
            if item.item_name and item.item_name.lower() != item.item_code.lower():
                item_data.append({
                    "item_name": item.item_code,  # Still use item code as identifier
                    "match_text": re.sub(r'[\[\]]', '', item.item_name.lower()),
                    "type": "name"
                })
        
        return item_data
    
    def extract_bracket_text(self, description):
        """Extract text within square brackets"""
        matches = re.findall(r'\[(.*?)\]', description)
        return matches[0] if matches else None
    
    def fuzzy_match_item(self, text, all_items):
        """Find the best item match using fuzzy matching"""
        if not text:
            return None
            
        # Clean text by removing special characters and brackets
        clean_text = re.sub(r'[\[\]]', '', text).lower().strip()
        best_match = None
        best_score = 0
        
        # Skip numeric-only strings
        if re.match(r'^\d+$', clean_text):
            return None
            
        for item in all_items:
            # Clean match text similarly
            clean_match = item["match_text"]
            
            # Calculate similarity score
            score = difflib.SequenceMatcher(None, clean_text, clean_match).ratio() * 100
            
            # Give extra weight to code matches
            if item["type"] == "code":
                score = min(score * 1.2, 100)  # Boost code matches by 20%
                
            if score > best_score:
                best_score = score
                best_match = {
                    "item_name": item["item_name"],  # Actual item code
                    "score": score,
                    "match_type": item["type"],
                    "match_text": item["match_text"]
                }
        
        # Return match only if it meets minimum confidence
        if best_score > 70:
            return best_match
            
        # Try again with bracket extraction if first match failed
        bracket_text = self.extract_bracket_text(text)
        if bracket_text:
            return self.fuzzy_match_item(bracket_text, all_items)
            
        return None

    def fuzzy_match_party(self, party_name):
        """Fuzzy match party against existing parties"""
        if not party_name:
            return None
            
        clean_name = party_name.lower().strip()
        party_type = self.party_type
        
        # Get all parties of the specified type
        if party_type == "Customer":
            parties = frappe.get_all("Customer", fields=["name", "customer_name"])
            names = [p["customer_name"] for p in parties]
        else:
            parties = frappe.get_all("Supplier", fields=["name", "supplier_name"])
            names = [p["supplier_name"] for p in parties]
            
        # Find best match
        best_match = None
        best_score = 0
        
        for i, name in enumerate(names):
            score = difflib.SequenceMatcher(None, clean_name, name.lower()).ratio() * 100
            if score > best_score:
                best_score = score
                best_match = {
                    "name": parties[i]["name"],
                    "score": score,
                    "match_name": name
                }
        
        # Return match only if above confidence threshold
        return best_match if best_score > 80 else None


@frappe.whitelist()
def extract_invoice(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        result = doc.extract_invoice()
        return result
    except Exception as e:
        frappe.log_error(f"Extract invoice failed: {str(e)}", "Extract Invoice Error")
        return {"status": "error", "message": str(e)}


@frappe.whitelist()
def create_invoice(docname, submit_invoice=False):
    """Create invoice from the Create Invoice button"""
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        # Create draft invoice by default
        doc.create_invoice_from_child(submit_invoice=submit_invoice)
        return {
            "status": "success",
            "invoice_name": doc.invoice_reference,
            "doctype": doc.invoice_type,
            "status": doc.invoice_status
        }
    except Exception as e:
        frappe.log_error(f"Create invoice failed: {str(e)}", "Create Invoice Error")
        return {"status": "error", "message": str(e)}


# Debug method to test OCR safely
@frappe.whitelist()
def debug_ocr_preview(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        file_path = get_file_path(doc.file)

        def preprocess_image(pil_img):
            try:
                img = np.array(pil_img.convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(scaled)
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 10
                )
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.erode(thresh, kernel, iterations=1)
                return processed
            except Exception as e:
                frappe.log_error(f"Debug image processing failed: {str(e)}", "OCR Debug Error")
                return pil_img

        text = ""
        if file_path.endswith(".pdf"):
            images = convert_from_path(file_path, dpi=300)
            for img in images:
                processed = preprocess_image(img)
                text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
        else:
            img = Image.open(file_path)
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

        # Save to document for debugging
        doc.raw_ocr_text = text[:10000]
        doc.save()
        
        return text[:5000]  # Limit output to first 5000 characters
    except Exception as e:
        frappe.log_error(f"OCR debug failed: {str(e)}", "OCR Debug Error")
        return f"Error: {str(e)}"