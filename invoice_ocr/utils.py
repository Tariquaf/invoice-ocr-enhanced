import frappe
import json
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from PIL import Image

@frappe.whitelist()
def create_invoice(name):
    doc = frappe.get_doc("Invoice Upload", name)
    if not doc.extracted_data:
        frappe.throw("No OCR data extracted.")
    data = json.loads(doc.extracted_data)

    if doc.party_type == "Supplier":
        pi = frappe.new_doc("Purchase Invoice")
        pi.supplier = doc.party
    else:
        pi = frappe.new_doc("Sales Invoice")
        pi.customer = doc.party

    for item in data.get("items", []):
        pi.append("items", {
            "item_name": item.get("description"),
            "qty": item.get("qty"),
            "rate": item.get("rate"),
        })

    pi.posting_date = data.get("date")
    pi.insert(ignore_permissions=True)
    return {"doctype": pi.doctype, "name": pi.name}

# def extract_invoice_data(docname):
#     doc = frappe.get_doc("Invoice Upload", docname)
#     doc.ocr_status = "Processing"
#     doc.save()
#     frappe.db.commit()

#     try:
#         file_path = get_file_path(doc.file)
#         frappe.logger().info(f"[OCR] File path: {file_path}")
#         text = ""

#         if file_path.endswith(".pdf"):
#             images = convert_from_path(file_path)
#             for img in images:
#                 text += pytesseract.image_to_string(img)
#         else:
#             img = Image.open(file_path)
#             text = pytesseract.image_to_string(img)

#         frappe.logger().info(f"[OCR] Extracted Text:\n{text}")

#         invoice_data = {
#             "invoice_no": extract_keyword(text, ["Invoice#", "Invoice No", "Invoice Number"]),
#             "date": extract_keyword(text, ["Date"]),
#             "items": extract_items(text),
#             "total": extract_keyword(text, ["Total", "Amount Due"])
#         }

#         doc.ocr_status = "Extracted"
#         doc.extracted_data = json.dumps(invoice_data, indent=2)
#         doc.save()
#         frappe.db.commit()

#     except Exception:
#         doc.ocr_status = "Failed"
#         doc.save()
#         frappe.db.commit()
#         frappe.log_error(frappe.get_traceback(), "OCR Failed")


def extract_invoice_data(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.ocr_status = "Processing"
    doc.save()
    frappe.db.commit()

    try:
        # FORCED SAMPLE TEXT
        text = """
        INVOICE
        Invoice No: INV-2025-001
        Date: 2025-05-20
        Tramadol Tablet 100mg   10    50.00   500.00
        Paracetamol Syrup 250ml 5     80.00   400.00
        Vitamin D3 Drops        2     150.00  300.00
        Total:                          PKR 1200.00
        """

        invoice_data = {
            "invoice_no": extract_keyword(text, ["Invoice#", "Invoice No", "Invoice Number"]),
            "date": extract_keyword(text, ["Date"]),
            "items": extract_items(text),
            "total": extract_keyword(text, ["Total", "Amount Due"])
        }

        doc.ocr_status = "Extracted"
        doc.extracted_data = json.dumps(invoice_data, indent=2)
        doc.save()
        frappe.db.commit()

    except Exception:
        doc.ocr_status = "Failed"
        doc.save()
        frappe.db.commit()
        frappe.log_error(frappe.get_traceback(), "OCR Failed")


        
def extract_keyword(text, keys):
    for line in text.splitlines():
        for key in keys:
            if key.lower() in line.lower():
                return line.split()[-1]
    return ""


def extract_items(text):
    lines = text.splitlines()
    items = []
    for line in lines:
        if "Qty" in line or "Description" in line or "Rate" in line:
            continue
        if any(char.isdigit() for char in line):
            parts = line.split()
            if len(parts) >= 3:
                items.append({
                    "description": parts[0],
                    "qty": parts[1],
                    "rate": parts[2],
                })
    return items
