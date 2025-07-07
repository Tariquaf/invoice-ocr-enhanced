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
    "description": ["DESCRIPTION","PARTICULARS","ITEM","PRODUCT","GOODS","MATERIAL","وصف","البند","الصنف"],
    "uom":         ["UNIT","UOM","MEASURE","UNIT OF MEASURE","وحدة","الوحدة","مقياس"],
    "quantity":    ["NO","QTY","QUANTITY","COUNT","العدد","مقدار"],
    "price":       ["UNIT PRICE","PRICE","RATE","COST","UNIT COST","السعر","معدل","التكلفة"],
    "amount":      ["TOTAL","AMOUNT","LINE TOTAL","VALUE","المبلغ","القيمة","الإجمالي"]
}
N = len(HEADER_FIELD_KEYWORDS)

def flatten_vertical_table_dynamic(text):
    """
    Collapse N-line vertical tables into a single-line header plus one line per row,
    using HEADER_FIELD_KEYWORDS to detect each column in any order.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    flat = []
    i = 0

    while i + N <= len(lines):
        block = lines[i : i + N]
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
            i += N

            # collapse subsequent N-line rows until TOTAL encountered
            while i + N <= len(lines):
                row = lines[i : i + N]
                desc = row[mapping["description"]].strip().upper()
                if desc in ("TOTAL", "GRAND TOTAL"):
                    break
                row_line = "  ".join(row[mapping[f]] for f in HEADER_FIELD_KEYWORDS)
                flat.append(row_line)
                i += N

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
    Return text if it’s at least `min_chars` long; otherwise None.
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

    # find end line (any amount keyword at start)
    end = None
    for i, line in enumerate(lines[start+1:], start+1):
        if any(line.upper().startswith(kw) for kw in HEADER_FIELD_KEYWORDS["amount"]):
            end = i + 1
            break
    end = end or len(lines)

    return "\n".join(lines[start:end])

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
            # Roll back to Draft and mark OCR as Failed
            frappe.db.set_value("Invoice Upload", self.name, {
                 "docstatus": 0,
                 "ocr_status": "Failed"
             })
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
        clean_name = raw_party.split("\n", 1)[0].strip()
        clean_name = re.sub(r'[^\w\s&\.-]', '', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
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


        # 3. Fuzzy candidates from both tables
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

        # 4. No matches?
        if not candidates:
            return False

        # 5. Pick top scoring match
        best = max(candidates, key=lambda x: x["score"])
        self.party_type = best["doctype"]
        self.party      = best["match_name"]
        frappe.log_error(
            f"Fuzzy party match: '{clean_name}' → "
            f"{best['doctype']}/{best['name']} @ {best['score']:.0f}%",
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
            # ─── 1. Validate and resolve file path ───────────────────────────────
            if not self.file:
                frappe.throw("No file attached to this invoice. Please upload a file before extracting.")

            start_time = time.time()
            file_path = get_file_path(self.file)
            if not file_path or not os.path.exists(file_path):
                frappe.throw(f"Unable to locate the uploaded file: {self.file}")

            text = ""

            # ─── 2. Choose OCR method: embedded PDF text or image-based ─────────
            ocr_config = frappe.conf.get("ocr_config", "--psm 4 --oem 3 -l eng+urd")
            file_is_pdf = file_path.lower().endswith(".pdf")

            if file_is_pdf:
                # Try extracting embedded text first (faster and more accurate)
                pdf_text = extract_pdf_text(file_path)
                if pdf_text:
                    text = pdf_text
                else:
                    # No embedded text? Fallback to OCR on first 3 pages
                    images = convert_from_path(file_path, dpi=200, first_page=1, last_page=3)
                    for img in images:
                        if time.time() - start_time > 120:
                            frappe.throw("OCR timed out. Try using a smaller file or limit to fewer pages.")
                        processed = preprocess_image(img)
                        text += pytesseract.image_to_string(processed, config=ocr_config)
                        img.close()
            else:
                # Treat as a single image input
                img = Image.open(file_path)
                try:
                    processed_img = preprocess_single_image(img)
                    text = pytesseract.image_to_string(processed_img, config=ocr_config)
                finally:
                    img.close()

            if not text.strip():
                frappe.throw("OCR failed to extract any content. Ensure the document quality is good.")

            # ─── 3. Normalize document layout ───────────────────────────────────
            text = flatten_vertical_table_dynamic(text)
            table_text = extract_items_section(text)

            # ─── 4. Extract line items using structured patterns ────────────────
            items = self.extract_table_items(table_text)
            if not items:
                # If structured extraction fails, fallback to charge/item blocks
                items = self.extract_charges(text) or self.extract_items_fallback(text)

            if not items:
                frappe.throw("No items found in the invoice. Double-check OCR quality or template alignment.")

            # ─── 5. Extract header fields: dates, reference, source, rep ────────
            dates = self.extract_dates(text)
            self.invoice_date     = dates.get("invoice_date")     or self.invoice_date
            self.due_date         = dates.get("due_date")         or self.due_date
            self.delivery_date    = dates.get("delivery_date")    or self.delivery_date
            self.source           = self.extract_source(text)     or self.source
            self.reference        = self.extract_reference(text)  or self.reference
            self.representative_name = self.extract_representative(text) or self.representative_name

            # ─── 6. Bundle all extracted data ──────────────────────────────────
            extracted_data = {
                "items": items,
                "party": self.extract_party(text) or ""
            }

            # ─── 7. Assign matching party (Customer or Supplier) ────────────────
            raw_party = extracted_data.get("party", "").strip()
            if raw_party:
                found = self.find_and_assign_party(raw_party)
                if not found:
                    frappe.log_error(
                        f"Could not resolve Party for: '{raw_party}'",
                        "Invoice Upload: Party Matching"
                    )
            # ─── 7.5 Populate child table with extracted items ──────────────────────
            self.set("invoice_upload_item", [])  # Clear existing rows
            all_items = self.get_items_for_matching()

            for item in items:
                matched_item = self.fuzzy_match_item(item["description"], all_items)

                self.append("invoice_upload_item", {
                    "ocr_description": item["description"],
                    "qty": item.get("qty", 1),
                    "rate": item.get("rate", 0.0),
                    "uom": self.fuzzy_match_uom(item.get("uom")) if item.get("uom") else None,
                    "item": matched_item["item_name"] if matched_item else None
                })

            # ─── 8. Save OCR output & mark status ──────────────────────────────
            self.extracted_data = json.dumps(extracted_data, indent=2)
            self.ocr_status = "Extracted"
            self.save()

            frappe.msgprint("OCR Extraction completed ✅\nPlease review and confirm before submitting.")

            return {
                "status": "success",
                "items": items,
                "party": self.party
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

    import re

    def extract_party(self, text):
        """Extract the actual partner name from the invoice text (OCR-tolerant)"""

        # Normalize text spacing and remove carriage returns
        text = text.replace('\r', '').replace('  ', ' ')

        # Fix common OCR errors
        def clean_common_ocr_errors(s):
            s = s.replace(' lron ', ' Iron ')
            s = s.replace(' lron', ' Iron')  # Edge case without spacing
            return s

        # 1. Try a global match anywhere in the text
        partner_match = re.search(r'Partner\s*Name\s*[:\-]?\s*([A-Z][a-zA-Z\s&.,\-]+)', text, re.IGNORECASE)
        if partner_match:
            party = clean_common_ocr_errors(partner_match.group(1).strip())
            if party:
                return party

        # 2. Fallback: line-by-line scan for "Partner Name"
        for line in text.split('\n'):
            if "partner name" in line.lower():
                match = re.search(r'Partner\s*Name\s*[:\-]?\s*([A-Z][a-zA-Z\s&.,\-]+)', line, re.IGNORECASE)
                if match:
                    party = clean_common_ocr_errors(match.group(1).strip())
                    if party:
                        return party

        # 3. Other common labels (Customer, Supplier, Invoice To, etc.)
        party_labels = [
            r"Invoice\s*To", r"Bill\s*To", r"Sold\s*To", r"Customer", r"Client", r"Supplier", r"Vendor"
        ]
        for label in party_labels:
            pattern = re.compile(fr'{label}\s*[:\-]?\s*([^\n]+)', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                party = clean_common_ocr_errors(match.group(1).strip())
                party = re.sub(r'[^\w\s\-]$', '', party).strip()
                if party:
                    return party

        # 4. Look for name in square brackets
        bracket_match = re.search(r'\[([^\]]+)\]', text)
        if bracket_match:
            candidate = bracket_match.group(1).strip()
            if re.search(r'[a-zA-Z]', candidate) and ' ' in candidate and not re.search(r'\d{5,}', candidate):
                return clean_common_ocr_errors(candidate)

        # 5. Try to extract a name-like phrase near the top of the document
        top_section = text.split("Invoice Date:")[0] if "Invoice Date:" in text else text[:500]
        name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', top_section)
        if name_candidates:
            name_candidates.sort(key=len, reverse=True)
            return clean_common_ocr_errors(name_candidates[0])

        # 6. Scan for name near invoice ID
        title_match = re.search(r'Invoice\s+\w+[\/\-]?\d+[\/\-]?\d*', text, re.IGNORECASE)
        if title_match:
            start_pos = max(0, title_match.start() - 150)
            end_pos = min(len(text), title_match.end() + 150)
            context = text[start_pos:end_pos]
            name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
            if name_candidates:
                name_candidates.sort(key=len, reverse=True)
                return clean_common_ocr_errors(name_candidates[0])

        # 7. Urdu fallback (if OCR is multilingual)
        urdu_match = re.search(r'(پارٹنر\s*نام|نام\s*خریدار)\s*[:\-]?\s*(.+)', text)
        if urdu_match:
            return urdu_match.group(2).strip()

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

    def fuzzy_match_party(self, party_name, min_score=70):
        """Fuzzy match party; only accept if score ≥ min_score."""
        if not party_name:
            return None

        clean_name = re.sub(r'[^\w\s&\.-]', '', party_name).lower().strip()

        # Load parties (Customer or Supplier)…
        if self.party_type == "Customer":
            parties = frappe.get_all("Customer", ["name", "customer_name"])
            key = "customer_name"
        else:
            parties = frappe.get_all("Supplier", ["name", "supplier_name"])
            key = "supplier_name"

        best_match = None
        best_score = 0.0

        for p in parties:
            raw = p.get(key) or ""
            clean_db = re.sub(r'[^\w\s&\.-]', '', raw).lower().strip()
            score = difflib.SequenceMatcher(None, clean_name, clean_db).ratio() * 100

            if score > best_score:
                best_score = score
                best_match = {"name": p["name"], "score": score, "match_name": raw}

        # Only return if above your dynamic threshold
        return best_match if best_score >= min_score else None
    
    def extract_table_items(self, text):
        """
        Extract items from a horizontal table using HEADER_FIELD_KEYWORDS.
        """
        import re
        lines = text.splitlines()
        header_map = {}
        header_idx = None

        # 1) Find header row by matching synonyms
        for idx, line in enumerate(lines):
            parts = line.split("|") if "|" in line else re.split(r"\s{2,}", line)
            parts = [p.strip() for p in parts if p.strip()]

            col_map = {}
            for i, col in enumerate(parts):
                uc = col.upper()
                for field, kws in HEADER_FIELD_KEYWORDS.items():
                    if any(k in uc for k in kws):
                        col_map[field] = i

            if {"description","quantity","price","amount"}.issubset(col_map):
                header_map = col_map
                header_idx = idx
                break

        if header_idx is None:
            return []

        # 2) Parse each row below header
        items = []
        for line in lines[header_idx+1:]:
            if not line.strip() or re.match(r"(-{3,}|={3,})", line):
                break

            parts = line.split("|") if "|" in line else re.split(r"\s{2,}", line)
            parts = [p.strip() for p in parts]

            try:
                desc      = parts[header_map["description"]]
                qty       = float(parts[header_map["quantity"]].replace(",", ""))
                price_str = parts[header_map["price"]].replace(",", "")
                amt_str   = parts[header_map["amount"]].replace(",", "")

                rate = float(price_str) if price_str else (float(amt_str)/qty if qty else 0.0)
                uom  = parts[header_map["uom"]] if "uom" in header_map else None

                items.append({
                    "description": desc,
                    "qty": qty,
                    "rate": rate,
                    "uom": uom
                })
            except Exception:
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