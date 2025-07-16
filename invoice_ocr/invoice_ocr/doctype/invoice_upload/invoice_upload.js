// invoice_ocr/invoice_ocr/doctype/invoice_upload/invoice_upload.js

frappe.ui.form.on('Invoice Upload', {
  // Re-run refresh when file or party_type changes
  file(frm) {
    frm.refresh();
  },
  party_type(frm) {
    frm.refresh();
  },

  refresh(frm) {
    // 0) Clear old buttons
    frm.remove_custom_button(__('Extract from file'));
    frm.remove_custom_button(__('Re-scan Document'));

    // 0a) If invoice already created, show locked message
    if (frm.doc.invoice_created) {
      frappe.msgprint({
        title: __('Locked'),
        message: __('This upload already created an invoice and cannot be reprocessed.'),
        indicator: 'grey'
      });
      return;
    }

    // 1) Only in draft with file + party_type
    if (frm.doc.docstatus === 0
        && frm.doc.file
        && frm.doc.party_type
        && !frm.doc.invoice_created) {

      const alreadyExtracted = frm.doc.ocr_status === 'Extracted';
      const label = alreadyExtracted
        ? __('Re-scan Document')
        : __('Extract from file');

      // safe extension parse
      const ext = (frm.doc.file || '').split('.').pop().toLowerCase();

      const method = (ext === 'xls' || ext === 'xlsx')
        ? 'invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_excel'
        : 'invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_invoice';

      const btn = frm.add_custom_button(label, () => {
        // 2) Must save unsaved changes first
        const do_extract = () => {
          // disable save until done
          frm.disable_save();
          _show_spinner();
          _call_extraction(method, frm, alreadyExtracted);
        };

        if (frm.dirty()) {
          frm.save().then(do_extract).catch(() => {
            frappe.msgprint({
              title: __('Error'),
              message: __('Save failed. Please try again.'),
              indicator: 'red'
            });
          });
        } else {
          do_extract();
        }
      });

      // style and tooltip
      btn.removeClass('btn-xs btn-primary')
         .addClass(alreadyExtracted ? 'btn-warning btn-lg' : 'btn-default btn-lg')
         .css({
           'font-size': '14px',
           'padding': '8px 20px',
           'margin-left': '10px',
           'background-color': alreadyExtracted ? '#f0ad4e' : 'transparent',
           'border': '1px solid #ccc',
           'color': '#333'
         })
         .attr('title', alreadyExtracted
           ? __('Re-run OCR scan with updated file or party')
           : __('Start invoice extraction from the uploaded file'));
    }
  },
  
  // ====== ADDED BEFORE_SAVE HANDLER ======
  before_save: function(frm) {
    // Clear any previous refresh timers
    if (frm.refresh_after_save) {
      clearTimeout(frm.refresh_after_save);
    }
    
    // Set timer to refresh form after save completes
    frm.refresh_after_save = setTimeout(() => {
      frm.reload_doc();
    }, 1000);
  }
  // ====== END OF ADDED HANDLER ======
});

// Helper to call server extraction
function _call_extraction(method, frm, alreadyExtracted) {
  frappe.call({
    method,
    args: { docname: frm.doc.name }
  }).then(r => {
    $('#custom-spinner').remove();
    // re-enable save
    frm.enable_save();

    if (r.message && r.message.status === 'success') {
      frappe.msgprint(
        alreadyExtracted
          ? __('Document re-scanned successfully!')
          : __('Data extracted successfully!')
      );
      frm.reload_doc();
    } else if (r.message && r.message.status === 'error') {
      frappe.msgprint({
        title: __('Extraction Failed'),
        message: r.message.message || __('No items could be matched.'),
        indicator: 'orange'
      });
    } else {
      frappe.msgprint({
        title: __('Error'),
        message: __('Unexpected error during extraction.'),
        indicator: 'red'
      });
    }
  }).catch(() => {
    $('#custom-spinner').remove();
    frm.enable_save();
    frappe.msgprint({
      title: __('Error'),
      message: __('Extraction failed. Please try again.'),
      indicator: 'red'
    });
  });
}

// Spinner overlay helper
function _show_spinner() {
  if (!$('#custom-spinner').length) {
    $('body').append(`
      <div id="custom-spinner" style="
        position: fixed; inset: 0;
        background: rgba(255,255,255,0.8);
        display: flex; align-items: center; justify-content: center;
        z-index: 99999;">
        <div style="
          border: 6px solid #ececec;
          border-top: 6px solid #0b62a4;
          border-radius: 50%;
          width: 70px; height: 70px;
          animation: spin 1s linear infinite;">
        </div>
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