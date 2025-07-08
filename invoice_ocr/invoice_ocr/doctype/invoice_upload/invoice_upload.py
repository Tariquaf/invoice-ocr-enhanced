import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import traceback
import difflib
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image
from frappe.utils import add_days, get_url_to_form, nowdate
import time

# Dynamic vertical table header flattening & lookup keywords
HEADER_FIELD_KEYWORDS = {
    "description": ["DESCRIPTION", "PARTICULARS", "ITEM", "PRODUCT", "GOODS", "MATERIAL", "وصف", "البند", "الصنف"],
    "uom":         ["UNIT", "UOM", "MEASURE", "UNIT OF MEASURE", "وحدة", "الوحدة", "مقياس"],
    "quantity":    ["NO", "QTY", "QUANTITY", "COUNT", "العدد", "مقدار"],
    "price":       ["UNIT PRICE", "PRICE", "RATE", "COST", "UNIT COST", "السعر", "معدل", "التكلفة"],
    "amount":      ["TOTAL", "AMOUNT", "LINE TOTAL", "VALUE", "المبلغ", "القيمة", "الإجمالي"]
}

def flatten_vertical_table_dynamic(text):
    """
    Collapse N-line vertical tables into a single-line header plus one line per row,
    using HEADER_FIELD_KEYWORDS to detect each column in any order.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    flat = []
    i = 0

    while i + len(HEADER_FIELD_KEYWORDS) <= len(lines):
        block = lines[i : i + len(HEADER_FIELD_KEYWORDS)]
        mapping = {}
        # match each block line to a header field
        for idx, line in enumerate(block):
            upper = line.upper()
            for field, kws in HEADER_FIELD_KEYWORDS.items():
                if field in mapping:
                    continue
                if any(key in upper for key in kws):
                    mapping[field] = idx
                    break

        # if all headers found in this block, collapse header and rows
        if set(mapping.keys()) == set(HEADER_FIELD_KEYWORDS.keys()):
            # build header line
            header_line = "  ".join(block[mapping[f]] for f in HEADER_FIELD_KEYWORDS)
            flat.append(header_line)
            i += len(HEADER_FIELD_KEYWORDS)

            # collapse subsequent N-line rows until TOTAL encountered
            while i + len(HEADER_FIELD_KEYWORDS) <= len(lines):
                row = lines[i : i + len(HEADER_FIELD_KEYWORDS)]
                desc = row[mapping["description"]].strip().upper()
                if desc in ("TOTAL", "GRAND TOTAL"):
                    break
                row_line = "  ".join(row[mapping[f]] for f in HEADER_FIELD_KEYWORDS)
                flat.append(row_line)
                i += len(HEADER_FIELD_KEYWORDS)

            continue

        # not a vertical table header, copy line as-is
        flat.append(lines[i])
        i += 1

    # append any trailing lines
    flat.extend(lines[i:])
    return "\n".join(flat)

def extract_pdf_text(file_path, min_chars=100):
    """
    Try to pull embedded text from the first 3 pages of a PDF.
    Return text if it's at least `min_chars` long; otherwise None.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages[:3]:
            text += (page.extract_text() or "") + "\n"
        return text if len(text) >= min_chars else None
    except Exception:
        return None

def extract_items_section(text):
    """
    Return only the lines from the first description-synonym header
    through the final amount-synonym line (inclusive).
    """
    lines = text.splitlines()

    # find start line (any description keyword)
    start = None
    for i, line in enumerate(lines):
        if any(kw in line.upper() for kw in HEADER_FIELD_KEYWORDS["description"]):
            start = i
            break
    if start is None:
        return text

    # find end line (any amount keyword at start or common section ends)
    end = None
    for i, line in enumerate(lines[start+1:], start+1):
        line_upper = line.strip().upper()
        if (any(line_upper.startswith(kw) for kw in HEADER_FIELD_KEYWORDS["amount"]) or
            line_upper in ("PRODUCT", "PAYMENT TERMS", "PAYMENT COMMUNICATION") or
            any(kw in line_upper for kw in ["SN/LN", "شرح", "تفصیل", "پروڈکٹ"])):
            end = i
            break
    end = end or len(lines)

    # Filter out non-item lines
    filtered_lines = []
    for line in lines[start:end]:
        line_upper = line.upper()
        if not any(kw in line_upper for kw in ["PARTNER NAME", "ORDER DATE", "PURCHASE ORDER", 
                                              "SHIPPING ADDRESS", "INVOICE DATE", "DUE DATE"]):
            filtered_lines.append(line)
            
    return "\n".join(filtered_lines)

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

def safe_float(s):
    """Convert cleaned numeric string to float, or None on error."""
    try:
        if s is None or s.strip() == "":
            return None
            
        # Handle rupee symbols and commas
        s = str(s).replace('Rs.', '').replace(',', '').strip()
        
        # Handle trailing letters (e.g., "1250.000kg")
        s = re.sub(r'[a-zA-Z]+$', '', s)
        
        # Handle multiple decimal points
        if s.count('.') > 1:
            parts = s.split('.')
            s = parts[0] + '.' + ''.join(parts[1:])
            
        return float(s)
    except Exception:
        return None

class InvoiceUpload(Document):
   
    def on_submit(self):
        try:
            self.reload()
            # Create draft invoice on submit
            self.create_invoice_from_child()  # No parameters needed now
            # Make document non-editable
            frappe.db.set_value("Invoice Upload", self.name, "docstatus", 1)
        except Exception as e:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            error_message = f"Invoice Creation Failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "Invoice Creation Failed")
            frappe.throw(f"Invoice creation failed: {str(e)}")

    def find_and_assign_party(self, raw_party):
        """
        Clean raw_party, look for exact match in Customer/Supplier.
        Fallback to fuzzy match across both tables.
        Sets self.party_type and self.party.
        Returns True if a match was assigned, False otherwise.
        """
        # Clean party name - remove address information
        clean_name = re.split(r'\n|Partner Address|, Partner', raw_party, flags=re.IGNORECASE)[0]
        clean_name = re.sub(r'[^\w\s&\.-]', '', clean_name).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name)
        lower_name = clean_name.lower()

        # Exact match in Customer
        if frappe.db.exists("Customer", {"customer_name": clean_name}):
            self.party_type = "Customer"
            self.party = clean_name
            return True

        # Exact match in Supplier
        if frappe.db.exists("Supplier", {"supplier_name": clean_name}):
            self.party_type = "Supplier"
            self.party = clean_name
            return True

        # Fuzzy match both
        candidates = []
        for doctype, field in (("Customer", "customer_name"), ("Supplier", "supplier_name")):
            for r in frappe.get_all(doctype, [field, "name"]):
                raw_db = r.get(field) or ""
                db_clean = re.sub(r'[^\w\s&\.-]', '', raw_db).lower().strip()
                score = difflib.SequenceMatcher(None, lower_name, db_clean).ratio() * 100
                if score >= 70:
                    candidates.append({
                        "doctype": doctype,
                        "name": r["name"],
                        "match_name": raw_db,
                        "score": score
                    })

        if not candidates:
            return False

        best = max(candidates, key=lambda x: x["score"])
        self.party_type = best["doctype"]
        self.party = best["match_name"]

        frappe.log_error(
            f"Fuzzy party match: '{clean_name}' → {best['doctype']}/{best['name']} @ {best['score']:.0f}%",
            "Invoice Upload: Party Match"
        )
        return True

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
                    # Convert PIL image to OpenCV format
                    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    # Grayscale conversion
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

                    # Thresholding to improve OCR accuracy
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Convert back to PIL for pytesseract
                    processed_img = Image.fromarray(thresh)

                    # OCR on processed image
                    text = pytesseract.image_to_string(processed_img, config=ocr_config)
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
            
            # Extract party and set it to the document field
            party_name = self.extract_party(text)
            if party_name:
                party_match = self.fuzzy_match_party(party_name)
                if party_match:
                    self.party = party_match["name"]  # Set matched party to document field
                else:
                    self.party = None  # Clear existing party if no match
            else:
                self.party = None  # Clear existing party if none extracted
            
            # Save immediately to capture metadata
            self.save()
            # ===== END METADATA EXTRACTION =====
            
            # Extract items using our multi-strategy approach
            items = self.extract_items(text)
            
            # Handle case where no items were automatically found
            if not items:
                frappe.msgprint("No items found automatically. Please add items manually.", 
                              alert=True, indicator="orange")
                self.ocr_status = "Partially Extracted"
                self.save()
                return {
                    "status": "partial_success",
                    "message": "Metadata extracted but no items found. Please add items manually."
                }
            
            extracted_data = {
                "items": items,
                "party": party_name
            }
            self.extracted_data = json.dumps(extracted_data, indent=2)
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
            frappe.log_error(f"Extracted party name: {party_name}", "Party Extraction")
            if party_name:
                party_match = self.fuzzy_match_party(party_name)
                if party_match:
                    frappe.log_error(f"Matched party: {party_match['name']} with score {party_match['score']}", "Party Matching")
                    extracted_data["party"] = party_match["name"]
                else:
                    frappe.log_error(f"No match found for party: {party_name}", "Party Matching")
                    self.party = None
            else:
                frappe.log_error("No party name extracted", "Party Extraction")
                self.party = None

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

    def create_invoice_from_child(self):
        """Create invoice on document submit"""
        # Check if invoice already created
        if self.invoice_created:
            frappe.throw("Invoice already created for this document")
            
        # Create the appropriate invoice type
        if self.party and self.party.strip():
            if self.party_type == "Supplier":
                inv = frappe.new_doc("Purchase Invoice")
                inv.supplier = self.party
            else:
                inv = frappe.new_doc("Sales Invoice")
                inv.customer = self.party
        else:
            # Create a generic invoice if party is not specified
            if self.party_type == "Supplier":
                inv = frappe.new_doc("Purchase Invoice")
            else:
                inv = frappe.new_doc("Sales Invoice")
            
            frappe.msgprint("No party specified. Creating invoice with blank party field.", 
                        alert=True, indicator="orange")

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
            if self.invoice_date:
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
        
        # Save invoice as draft
        try:
            # Bypass validations for draft invoice creation
            inv.flags.ignore_validate = True
            inv.flags.ignore_mandatory = True
            inv.insert(ignore_permissions=True)
        except Exception as e:
            # Log error but don't prevent document submission
            error_msg = f"Invoice creation failed: {str(e)}"
            frappe.msgprint(error_msg, alert=True, indicator="red")
            frappe.log_error(error_msg, "Invoice Creation Error")
            raise  # Re-raise exception to prevent document submission
        
        # Update status and reference
        frappe.db.set_value(self.doctype, self.name, {
            "invoice_created": 1,
            "invoice_reference": inv.name,
            "invoice_type": inv.doctype,
            "invoice_status": "Draft"
        })

        # Create clickable link to the invoice
        invoice_url = get_url_to_form(inv.doctype, inv.name)
        frappe.msgprint(f"<a href='{invoice_url}'>{inv.name}</a> created (Draft)")

        # Add to activity with direct link
        comment_txt = f"Created {inv.doctype}: <a href='{invoice_url}'>{inv.name}</a>"
        self.add_comment('Info', comment_txt)
    
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

    def extract_fields_from_labeled_block(self, text):
        import re
        lines = text.splitlines()
        results = {}

        for i, line in enumerate(lines):
            if line.count(':') >= 2 and i + 1 < len(lines):
                titles = [t.strip() for t in line.split(':') if t.strip()]
                values_line = lines[i + 1].strip()
                values = values_line.split()

                if len(titles) == 2 and len(values) >= 2:
                    for vi, v in enumerate(values):
                        if re.match(r'\d{2}/\d{2}/\d{4}', v):
                            rep_name = " ".join(values[:vi])
                            order_date = v + " " + " ".join(values[vi+1:])
                            results[titles[0]] = rep_name
                            results[titles[1]] = order_date
                            break

        return results
    
    def extract_representative(self, text):
        """Extract representative name from invoice (corrected with next-line logic)"""
        try:
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
                    name = re.sub(r'\s+\d+.*$', '', name)
                    name = re.sub(r'[^\w\s\.-]', '', name).strip()
                    if name and len(name) > 3:
                        return name

            # Fallback: look for "Purchase Representative" and skip misleading inline labels
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "purchase representative" in line.lower():
                    # If the line contains "Order Deadline", we skip extracting from this line
                    if "order deadline" in line.lower():
                        # Go to next line
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            name = re.sub(r'\s+\d+.*$', '', next_line)
                            name = re.sub(r'[^\w\s\.-]', '', name).strip()
                            if name and len(name) > 3 and not re.search(r'\d', name):
                                return name
                    else:
                        # Safe to try extracting from the same line
                        parts = re.split(r':', line)
                        if len(parts) > 1:
                            name = parts[1].strip()
                            name = re.sub(r'\s+\d+.*$', '', name)
                            name = re.sub(r'[^\w\s\.-]', '', name).strip()
                            if name and len(name) > 3 and not re.search(r'\d', name):
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
        # First try enhanced table extraction
        table_items = self.extract_table_items_enhanced(text)
        if table_items:
            return table_items

        # Try fallback table extraction
        table_items = self.extract_table_items_fallback(text)
        if table_items:
            return table_items

        # Then try bill of charges extraction
        charge_items = self.extract_charges(text)
        if charge_items:
            return charge_items

        # Finally, use context-aware extraction
        return self.extract_items_context_aware(text)

    def extract_table_items_enhanced(self, text):
        """Enhanced table extraction with multi-strategy detection"""
        items = []
        lines = text.splitlines()
        
        # Strategy 1: Header-based detection
        header_patterns = [
            r"DESCRIPTION.*QUANTITY.*UOM.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*UOM.*RATE.*AMOUNT",
            r"ITEM.*QTY.*UOM.*PRICE.*TOTAL",
            r"DESCRIPTION.*QUANTITY.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*RATE.*AMOUNT",
            r"ITEM.*QTY.*PRICE.*TOTAL",
            r"PRODUCT.*QUANTITY.*PRICE.*TOTAL",
            r"GOODS.*QTY.*RATE.*AMOUNT",
            r"ARTICLES.*QUANTITY.*PRICE.*TOTAL"
        ]
        
        # Find header line
        start_index = -1
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    start_index = i + 1
                    break
            if start_index != -1:
                break
        
        # Strategy 2: Amount alignment detection
        if start_index == -1:
            amount_positions = []
            for i, line in enumerate(lines):
                if re.search(r'\d+\.\d{2,3}$', line):
                    amount_positions.append(i)
            
            # Look for clustered amounts (likely a table)
            if len(amount_positions) > 3:
                start_index = max(0, amount_positions[0] - 5)
        
        # Strategy 3: Item code pattern detection
        if start_index == -1:
            for i, line in enumerate(lines):
                if re.match(r'^[A-Z]{3,}\d{3,}', line) or re.match(r'^\d+\.\s+\w+', line):
                    start_index = i
                    break
        
        if start_index == -1:
            return items
        
        # Process lines from start_index to end or totals
        for i in range(start_index, len(lines)):
            line = lines[i].strip()
            
            # Stop conditions
            if not line:
                continue
            if "TOTAL" in line.upper() or "GRAND" in line.upper() or "BALANCE" in line.upper():
                break
            if any(word in line.upper() for word in ["PAGE", "INVOICE", "DATE", "TERMS", "THANK"]):
                continue
                
            # Try multiple splitting strategies
            parts = []
            if '|' in line:  # Pipe-delimited
                parts = [p.strip() for p in line.split('|')]
            elif '\t' in line:  # Tab-delimited
                parts = [p.strip() for p in line.split('\t')]
            else:  # Space-delimited
                parts = re.split(r'\s{3,}', line)  # Split on 3+ spaces
            
            # Need at least 3 parts (description, qty, price)
            if len(parts) < 3:
                continue
                
            try:
                # Extract description (first columns except last 2)
                description = " ".join(parts[:-2]).strip()
                
                # If description is too short, use first column only
                if len(description) < 3:
                    description = parts[0].strip()
                    qty_str = parts[1] if len(parts) > 1 else ""
                    rate_str = parts[2] if len(parts) > 2 else ""
                else:
                    qty_str = parts[-2] if len(parts) >= 2 else ""
                    rate_str = parts[-1] if len(parts) >= 1 else ""
                
                # Skip header-looking lines
                if "DESCRIPTION" in description.upper() or "QUANTITY" in description.upper():
                    continue
                    
                # Clean numeric values
                qty_str = re.sub(r'[^\d\.]', '', qty_str.replace(',', ''))
                rate_str = re.sub(r'[^\d\.]', '', rate_str.replace(',', ''))
                
                # Skip if we can't parse numbers
                if not qty_str or not rate_str:
                    continue
                    
                qty = float(qty_str)
                rate = float(rate_str)
                
                # Skip zero quantities or rates
                if qty == 0 or rate == 0:
                    continue
                    
                # Clean description
                description = re.sub(r'[^\w\s\-\.]', '', description)
                description = re.sub(r'\s+', ' ', description).strip()
                
                # Skip invalid descriptions
                if len(description) < 3 or description.isdigit():
                    continue
                    
                # Extract UOM if available (likely after qty)
                uom = None
                if len(parts) >= 3:
                    uom_candidate = parts[-3].strip()
                    if re.match(r'^[A-Za-z]{1,5}$', uom_candidate):
                        uom = uom_candidate
                
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom
                })
                
            except Exception as e:
                # Skip lines we can't parse
                continue
        
        return items

    def extract_table_items_fallback(self, text):
        """Fallback table extraction for poorly structured tables"""
        items = []
        lines = text.splitlines()
        
        # Look for item lines with common patterns
        item_pattern = r'(\d+)\s+([A-Za-z]{1,5})?\s+([\d,]+\.\d{2,3})'
        
        for line in lines:
            match = re.search(item_pattern, line)
            if not match:
                continue
                
            try:
                qty = float(match.group(1).replace(',', ''))
                uom = match.group(2) if match.group(2) else None
                rate = float(match.group(3).replace(',', ''))
                
                # Get description (text before the pattern)
                desc_start = line.find(match.group(0))
                description = line[:desc_start].strip()
                
                # Clean description
                description = re.sub(r'[^\w\s\-\.]', '', description)
                description = re.sub(r'\s+', ' ', description).strip()
                
                # Skip invalid descriptions
                if len(description) < 3 or description.isdigit():
                    continue
                    
                # Skip zero quantities or rates
                if qty == 0 or rate == 0:
                    continue
                    
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom
                })
            except:
                continue
        
        return items

    def extract_items_context_aware(self, text):
        """Context-aware extraction with improved patterns"""
        items = []
        
        # Find the start of items section
        start_index = 0
        section_keywords = ["DESCRIPTION", "PARTICULARS", "PRODUCT", "ITEM", "GOODS"]
        for keyword in section_keywords:
            idx = text.find(keyword)
            if idx != -1:
                start_index = idx + len(keyword)
                break
        
        # Find the end of items section
        end_index = len(text)
        end_keywords = ["TOTAL", "SUBTOTAL", "GRAND TOTAL", "BALANCE", "PAYMENT"]
        for keyword in end_keywords:
            idx = text.find(keyword, start_index)
            if idx != -1:
                end_index = idx
                break
        
        # Focus only on the item section
        item_section = text[start_index:end_index]
        
        # Look for quantity patterns in the item section
        pattern = r'(\d+\.\d{3}|\d+)\s*([a-zA-Z]{1,5})?\s+(\d+\.\d{2,3})'
        matches = list(re.finditer(pattern, item_section))
        
        for match in matches:
            try:
                qty_str = match.group(1).replace(',', '')
                qty = float(qty_str)
                uom = match.group(2) if match.group(2) else None
                rate_str = match.group(3).replace(',', '')
                rate = float(rate_str)
                
                # Get description context
                line_start = item_section.rfind('\n', 0, match.start()) + 1
                line_end = item_section.find('\n', match.end())
                full_line = item_section[line_start:line_end].strip()
                
                # Extract description (remove any trailing numbers/symbols)
                description = full_line.split(match.group(0))[0].strip()
                description = re.sub(r'[\d\W]+$', '', description).strip()
                
                # Skip short descriptions
                if len(description) < 3 or description.isdigit():
                    continue
                    
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom
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
        """Fuzzy match UOM text to standard UOMs with OCR error mapping"""
        if not uom_text:
            return None
            
        # Map common OCR errors
        uom_map = {
            "kg": "Kg", "kgs": "Kg", "kg.": "Kg", "kilo": "Kg", "kilogram": "Kg",
            "units": "Unit", "unit": "Unit", "unts": "Unit", "un": "Unit", "nos": "Nos",
            "pcs": "Pcs", "pc": "Pcs", "pieces": "Pcs", "bags": "Bag", "bag": "Bag"
        }
        
        clean_text = uom_text.lower().strip()
        if clean_text in uom_map:
            return uom_map[clean_text]
            
        # Skip long strings that can't be UOM
        if len(clean_text) > 15:
            return None
            
        uoms = frappe.get_all("UOM", fields=["name"])
        
        best_match = None
        best_score = 0
        
        for uom in uoms:
            # Skip long UOMs
            if len(uom.name) > 15:
                continue
                
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
        """
        Extract PO/SO (or RFQ) only from the invoice header,
        ignoring anything in the items table below.
        """
        import re

        # 1) Limit search to header (everything before first "DESCRIPTION")
        upper = text.upper()
        idx = upper.find("DESCRIPTION")
        header = text[:idx] if idx != -1 else text

        # 2) Try RFQ pattern first
        rfq = re.search(
            r'Request\s*for\s*Quotation\s*[#:]*\s*([A-Z0-9-]+)',
            header, re.IGNORECASE
        )
        if rfq:
            return rfq.group(1).strip()

        # 3) Look for PO/SO patterns in header
        patterns = [
            r'\bPO[-\s]*([A-Z0-9-]{5,})\b',
            r'\bSO[-\s]*([A-Z0-9-]{5,})\b',
            r'Purchase\s*Order[-:\s]*([A-Z0-9-]+)',
            r'Sales\s*Order[-:\s]*([A-Z0-9-]+)',
            r'Order\s*Number[-:\s]*([A-Z0-9-]+)'
        ]
        for p in patterns:
            m = re.search(p, header, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # 4) Fallback: first standalone all-caps+digits code in header
        standalone = re.findall(r'\b[A-Z]{2,}\d{2,}\b', header)
        if standalone:
            return standalone[0]

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
        """Robust party extraction with OCR error tolerance"""
        # Normalize text for better matching
        clean_text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        clean_text = re.sub(r'[^\w\s\-\'&.,]', '', clean_text)  # Remove special chars but keep essential punctuation
        
        # 1. Direct label matching with OCR-tolerant patterns
        patterns = [
            r'Partner\s*Name\s*[:\-]?\s*([^\n\d]{10,50})',  # Increased length tolerance
            r'Customer\s*Name\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Supplier\s*Name\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Bill\s*To\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Sold\s*To\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Invoice\s*To\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Client\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Vendor\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Supplier\s*[:\-]?\s*([^\n\d]{10,50})',
            r'Customer\s*[:\-]?\s*([^\n\d]{10,50})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Remove trailing invalid characters
                name = re.sub(r'[^\w\s\-&.,]+$', '', name)
                if len(name) > 5:  # Reasonable length check
                    return name
        
        # 2. Multi-line extraction for better OCR tolerance
        lines = clean_text.split('\n')
        label_indices = []
        labels = ["partner name", "customer name", "supplier name", 
                 "bill to", "sold to", "invoice to", "client", "vendor"]
        
        # Find all possible label positions
        for i, line in enumerate(lines):
            if any(label in line.lower() for label in labels):
                label_indices.append(i)
        
        # Check next line after each label
        for idx in label_indices:
            if idx + 1 < len(lines):
                candidate = lines[idx+1].strip()
                # Basic validation - should contain letters and spaces
                if re.search(r'[a-zA-Z]{3}', candidate) and not re.search(r'\d{5,}', candidate):
                    return candidate
        
        # 3. Look for name in square brackets (common in invoices)
        bracket_match = re.search(r'\[([^\]]+)\]', clean_text)
        if bracket_match:
            candidate = bracket_match.group(1).strip()
            if re.search(r'[a-zA-Z]', candidate) and ' ' in candidate:
                return candidate
        
        # 4. Top section name extraction
        top_section = clean_text[:1000]  # First 1000 characters
        # Look for uppercase words that might be a company name
        name_candidates = re.findall(r'\b(?:[A-Z][a-z]+\s+){1,5}(?:Ltd|LLC|Inc|Co|GmbH|Pvt)\b', top_section)
        if name_candidates:
            return name_candidates[0]
        
        # 5. Near invoice number pattern
        inv_match = re.search(r'Invoice\s+[A-Z0-9/-]+', clean_text, re.IGNORECASE)
        if inv_match:
            start = max(0, inv_match.start() - 100)
            end = min(len(clean_text), inv_match.end() + 100)
            context = clean_text[start:end]
            name_candidates = re.findall(r'\b(?:[A-Z][a-z]+\s+){2,}\b', context)
            if name_candidates:
                # Return the longest candidate
                return max(name_candidates, key=len)
        
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
    
    def validate(self):
        if self.extracted_data:
            fields = self.extract_fields_from_labeled_block(self.extracted_data)
            
            # Define a mapping from extracted labels to DocType field names
            field_map = {
                "Purchase Representative": "representative_name",
         #       "Order Deadline": "order_deadline",
                # Add more mappings here if needed
            }
            
            for label, fieldname in field_map.items():
                setattr(self, fieldname, fields.get(label))

    def fuzzy_match_party(self, party_name):
        """Fuzzy match with improved handling for OCR artifacts"""
        if not party_name:
            return None
            
        # Clean name for matching - remove common OCR artifacts
        clean_name = re.sub(r'(?i)\b(partner|address|invoice|inv|sb|,)\b', '', party_name)
        clean_name = re.sub(r'[^\w\s]', '', clean_name).strip()
        
        if not clean_name:
            return None
            
        party_type = self.party_type
        
        # Get all parties of the specified type
        if party_type == "Customer":
            parties = frappe.get_all("Customer", fields=["name", "customer_name"])
            names = [p["customer_name"] for p in parties]
        else:
            parties = frappe.get_all("Supplier", fields=["name", "supplier_name"])
            names = [p["supplier_name"] for p in parties]
            
        # Find best match with multiple strategies
        best_match = None
        best_score = 0
        
        # Strategy 1: Full name matching
        for i, name in enumerate(names):
            clean_db_name = re.sub(r'[^\w\s]', '', name).lower()
            score = difflib.SequenceMatcher(None, clean_name.lower(), clean_db_name).ratio() * 100
            if score > best_score:
                best_score = score
                best_match = {
                    "name": parties[i]["name"],
                    "score": score,
                    "match_name": name
                }
        
        # Accept if we have a good match
        if best_score > 80:
            return best_match
            
        # Strategy 2: Word-based matching
        clean_words = set(clean_name.lower().split())
        for i, name in enumerate(names):
            clean_db_name = re.sub(r'[^\w\s]', '', name).lower()
            db_words = set(clean_db_name.split())
            
            # Calculate Jaccard similarity
            intersection = clean_words & db_words
            union = clean_words | db_words
            score = len(intersection) / len(union) * 100 if union else 0
            
            if score > best_score:
                best_score = score
                best_match = {
                    "name": parties[i]["name"],
                    "score": score,
                    "match_name": name
                }
        
        # Only return if we have a reasonable match
        return best_match if best_score > 60 else None
    
    def extract_table_items(self, text):
        """Extract items from structured tables with comprehensive scanning"""
        items = []
        lines = text.splitlines()
        
        # Find the start of the items table using more flexible patterns
        start_index = -1
        header_patterns = [
            r"DESCRIPTION.*QUANTITY.*UOM.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*UOM.*RATE.*AMOUNT",
            r"ITEM.*QTY.*UOM.*PRICE.*TOTAL",
            r"DESCRIPTION.*QUANTITY.*UNIT PRICE.*AMOUNT",
            r"PARTICULARS.*QUANTITY.*RATE.*AMOUNT",
            r"ITEM.*QTY.*PRICE.*TOTAL",
            r"PRODUCT.*QUANTITY.*PRICE.*TOTAL",
            r"GOODS.*QTY.*RATE.*AMOUNT",
            r"ARTICLES.*QUANTITY.*PRICE.*TOTAL"
        ]
        
        # Find the header line
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    start_index = i + 1
                    break
            if start_index != -1:
                break
        
        # If no header found, try to find table by common prefixes
        if start_index == -1:
            for i, line in enumerate(lines):
                if re.match(r'^\d+\s+\w+', line) or re.match(r'^[A-Z]{3,}\d+', line):
                    start_index = i
                    break
        
        # If still no start found, return empty
        if start_index == -1:
            return items
        
        # Process lines until we hit totals or end of table
        in_table = True
        i = start_index
        while in_table and i < len(lines):
            line = lines[i].strip()
            i += 1
            
            # Skip empty lines and separators
            if not line or line.startswith(('---', '===', '___', '***')):
                continue
                
            # Stop at totals section
            if "TOTAL" in line.upper() or "GRAND TOTAL" in line.upper() or "SUBTOTAL" in line.upper():
                break
                
            # Skip lines that are clearly headers or footers
            if any(word in line.upper() for word in ["PAGE", "INVOICE", "DATE", "TERMS"]):
                continue
                
            # Split line by common separators
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
            elif '\t' in line:
                parts = [part.strip() for part in line.split('\t')]
            else:
                parts = re.split(r'\s{3,}', line)  # Split on 3+ spaces
                
            # Need at least description, quantity and rate
            if len(parts) < 3:
                continue
                
            try:
                # Extract description (first columns except last 3)
                description = " ".join(parts[:-3]).strip()
                
                # If description is too short, use first column only
                if len(description) < 3:
                    description = parts[0].strip()
                    qty_str = parts[1] if len(parts) > 1 else ""
                    rate_str = parts[2] if len(parts) > 2 else ""
                else:
                    qty_str = parts[-3] if len(parts) >= 3 else ""
                    rate_str = parts[-2] if len(parts) >= 3 else ""
                
                # Clean numeric values
                qty_str = re.sub(r'[^\d\.]', '', qty_str.replace(',', ''))
                rate_str = re.sub(r'[^\d\.]', '', rate_str.replace(',', ''))
                
                # Skip if we can't parse numbers
                if not qty_str or not rate_str:
                    continue
                    
                qty = float(qty_str)
                rate = float(rate_str)
                
                # Clean description
                description = re.sub(r'[^\w\s\-\.]', '', description)
                description = re.sub(r'\s+', ' ', description).strip()
                
                # Skip invalid descriptions
                if len(description) < 3:
                    continue
                    
                # Extract UOM if available (second last column)
                uom = None
                if len(parts) >= 4:
                    uom_candidate = parts[-4].strip()
                    if re.match(r'^[A-Za-z]{1,5}$', uom_candidate):
                        uom = uom_candidate
                
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom
                })
                
            except Exception as e:
                # Skip lines we can't parse
                continue
        
        return items

@frappe.whitelist()
def extract_invoice(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        result = doc.extract_invoice()
        return result
    except Exception as e:
        frappe.log_error(f"Extract invoice failed: {str(e)}", "Extract Invoice Error")
        return {"status": "error", "message": str(e)}

# Debug method to test OCR safely
@frappe.whitelist()
def debug_ocr_preview(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        file_path = get_file_path(doc.file)

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

            # ✅ ADD THIS BLOCK HERE:
        with open("output.txt", "w") as f:
            f.write(text)
        import os
        print("Saved OCR output to:", os.path.abspath("output.txt"))

        # Save to document for debugging
        doc.raw_ocr_text = text[:10000]
        doc.save()
        
        return text[:5000]  # Limit output to first 5000 characters
    except Exception as e:
        frappe.log_error(f"OCR debug failed: {str(e)}", "OCR Debug Error")
        return f"Error: {str(e)}"