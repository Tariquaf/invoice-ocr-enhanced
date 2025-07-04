# 📄 Invoice OCR (Enhanced) App for ERPNext v15 

This repository offers a significantly improved version of the Invoice OCR tool, originally created by [Mohtashim-1](https://github.com/Mohtashim-1/Invoice-OCR). It automates data extraction from scanned PDF or image invoices and creates Sales or Purchase Invoices in ERPNext using OCR. This module was optimised for Odoo default Vendor Bills and Customer Invoices especially if you want to migrate from Odoo to ERPNext and wants to keep all sales/purchase data. Even if your vendor and/or your customer is using Odoo, you can scan bills and invoices.

I shall be adding more format of invoices in future if required so. Contributions are welcomed. 

## 🚀 What's New in This Fork

- 🔁 Support for both Tesseract and optional PaddleOCR
- 🧾 Smarter parsing with improved line-item and metadata extraction
- 🌍 Multi-language OCR (including Urdu, Arabic, etc.)
- 🖼️ Advanced image preprocessing (deskewing, denoising, etc.)
- 📤 Flexible export options (JSON, CSV, or direct ERPNext integration)
- 🧱 Modular, extensible codebase with better logging and error handling

## 🧠 Core Logic Overview

The script defines a Frappe DocType `InvoiceUpload` that:

- Extracts text from uploaded PDFs or images using `pytesseract`
- Parses invoice data: number, date, totals, line-items
- Auto-creates draft Sales/Purchase Invoices in ERPNext
- Handles file processing errors with graceful logging

## 📂 ERPNext DocType: Invoice Upload

| Field            | Description                                |
|------------------|--------------------------------------------|
| Party Type       | Customer / Supplier                        |
| File             | Attach scanned invoice                     |
| OCR Status       | Pending / Processing / Extracted / Failed  |
| Extracted Data   | Raw JSON of OCR output                     |
| Create Invoice   | Triggers invoice generation in ERPNext     |

---

## ⚙️ Full Installation Guide & How to Use

### ✅ 1. Prerequisites

Install required system packages:

```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-urd
sudo apt install poppler-utils  # For PDF processing
sudo apt install libgl1-mesa-glx  # For OpenCV according to your distribution

## Get the app from GitHub

```bash
bench get-app https://github.com/Tariquaf/invoice-ocr-enhanced.git

## Activate your Frappe virtual environment

```bash
source ~/frappe-bench/env/bin/activate

## Install required Python libraries

```bash
pip install -r ~/frappe-bench/apps/invoice_ocr/requirements.txt

# Or manually install requirements

pip install opencv-python-headless pytesseract numpy PyPDF2 pdf2image Pillow requests

## Verify dependencies

```bash
python3 ~/frappe-bench/apps/invoice_ocr/verify_dep.py

## Deactivate virtual enviroment

```bash
deactivate

## 4. Install the app on your site

```bash
cd ~/frappe-bench
```bash
bench --site yoursite.com install-app invoice_ocr

## Apply necessary migrations
```bash
bench migrate

## Restart bench or supervisor
```bash
bench restart #for production
bench start #for development

### How to use
- From awsome bar, search for "New Invoice Upload"
- Select Customer or Supplier depending upon invoice type
- Click attach button and attach/select invoice
- A button "Extract from File" will appear on top
- Save and submit after verification. It will create a draft invoice and further amendments can be made in draft invoice.
