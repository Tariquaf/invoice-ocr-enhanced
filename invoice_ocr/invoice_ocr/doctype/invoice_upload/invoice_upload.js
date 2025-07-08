// Copyright (c) 2025, Tariq and contributors
// For license information, please see license.txt

frappe.ui.form.on('Invoice Upload', {
    refresh(frm) {
        if (!frm.is_new() && frm.doc.file) {
            let $btn = frm.add_custom_button(
                __('Extract from file'),
                () => {
                    // spinner overlay
                    const spinnerHtml = `
                        <div id="custom-spinner" style="
                            position: fixed;
                            inset: 0;
                            background: rgba(255,255,255,0.8);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            z-index: 199999;
                        ">
                            <div style="
                                border: 6px solid #ececec;
                                border-top: 6px solid #0b62a4;
                                border-radius: 50%;
                                width: 70px;
                                height: 70px;
                                animation: spin 1s linear infinite;
                            "></div>
                        </div>
                    `;
                    $('body').append(spinnerHtml);
                    if (!$('#spinner-style').length) {
                        $('<style id="spinner-style">')
                            .text(`
                                @keyframes spin {
                                  0%   { transform: rotate(0deg); }
                                  100% { transform: rotate(360deg); }
                                }
                            `).appendTo('head');
                    }
                    frappe.call({
                        method: "invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_invoice",
                        args: { docname: frm.doc.name },
                        callback: () => {
                            $('#custom-spinner').remove();
                            frm.reload_doc();
                        },
                        error: () => {
                            $('#custom-spinner').remove();
                            frappe.msgprint({
                                title: __('Error'),
                                indicator: 'red',
                                message: __('Extraction failed. Please try again.')
                            });
                        }
                    });
                }
            );

            // remove black background: use default (white) style
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
