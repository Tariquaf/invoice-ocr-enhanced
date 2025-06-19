// Copyright (c) 2025, mohtashim and contributors
// For license information, please see license.txt

frappe.ui.form.on('Invoice Upload', {
    create_invoice: function (frm) {
        frappe.call({
            method: "invoice_ocr.utils.create_invoice",
            args: { name: frm.doc.name },
            callback: function (r) {
                if (r.message) {
                    frappe.model.sync(r.message);
                    frappe.set_route("Form", r.message.doctype, r.message.name);
                }
            }
        });
    }
});

frappe.ui.form.on("Invoice Upload", {
  refresh(frm) {
    if (!frm.is_new() && frm.doc.ocr_status !== "Extracted") {
      frm.add_custom_button("Extract from File", function () {
        frappe.call({
        method: "invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_invoice",
        args: { docname: frm.doc.name },
        callback: function (r) {
          if (r.message) {
            frappe.model.sync(r.message);
            frappe.set_route("Form", r.message.doctype, r.message.name);
          }
          frm.reload_doc();
        }
        });
      });
    }
  }
});