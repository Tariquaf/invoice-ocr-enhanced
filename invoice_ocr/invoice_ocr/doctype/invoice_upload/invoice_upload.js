// invoice_ocr/invoice_ocr/doctype/invoice_upload/invoice_upload.js
// Copyright (c) 2025, Tariq and contributors
// For license information, please see license.txt

frappe.ui.form.on('Invoice Upload', {
  // Prevent Submit until OCR has completed
  before_submit(frm) {
    if (frm.doc.ocr_status !== 'Extracted') {
      frappe.msgprint({
        title: __('Extraction Required'),
        message: __('Please click “Extract from file” and wait until OCR status is “Extracted” before submitting.'),
        indicator: 'orange'
      });
      frappe.validated = false;
    }
  },

  // Runs on form load & refresh
  refresh(frm) {
    // 1. Remove any existing Extract button
    frm.remove_custom_button(__('Extract from file'));

    // 2. Only show Extract button when:
    //    • draft (docstatus = 0)
    //    • a file is attached
    //    • OCR not done yet
    if (frm.doc.docstatus === 0
        && frm.doc.file
        && frm.doc.ocr_status !== 'Extracted') {

      let $btn = frm.add_custom_button(__('Extract from file'), () => {
        _show_spinner();

        // pick correct server method by file extension
        const ext = frm.doc.file.split('.').pop().toLowerCase();
        const method = (ext === 'xlsx' || ext === 'xls')
          ? 'invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_excel'
          : 'invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_invoice';

        frappe.call({
          method: method,
          args: { docname: frm.doc.name },
          callback: (r) => {
            $('#custom-spinner').remove();

            if (r.message && r.message.status === 'success') {
              frappe.msgprint(__('Data extracted successfully!'));
              frm.reload_doc();
            } else {
              const msg = (r.message && r.message.message) || __('Unexpected error');
              frappe.msgprint({
                title: __('Error'),
                message: msg,
                indicator: 'red'
              });
            }
          },
          error: () => {
            $('#custom-spinner').remove();
            frappe.msgprint({
              title: __('Error'),
              message: __('Extraction failed. Please try again.'),
              indicator: 'red'
            });
          }
        });
      });

      // 3. Style the Extract button
      $btn
        .removeClass('btn-xs btn-primary')
        .addClass('btn-default btn-lg')
        .css({
          'font-size': '14px',
          'padding': '8px 20px',
          'margin-left': '10px',
          'background-color': 'transparent',
          'border': '1px solid #ccc',
          'color': '#333'
        });
    }
  }
});

// Spinner overlay helper
function _show_spinner() {
  if (!$('#custom-spinner').length) {
    $('body').append(`
      <div id="custom-spinner" style="
        position: fixed; inset: 0;
        background: rgba(255,255,255,0.8);
        display: flex; align-items: center; justify-content: center;
        z-index: 99999;
      ">
        <div style="
          border: 6px solid #ececec;
          border-top: 6px solid #0b62a4;
          border-radius: 50%;
          width: 70px; height: 70px;
          animation: spin 1s linear infinite;
        "></div>
      </div>
    `);

    if (!$('#spinner-style').length) {
      $('<style id="spinner-style">').text(`
        @keyframes spin {
          0%   { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `).appendTo('head');
    }
  }
}
