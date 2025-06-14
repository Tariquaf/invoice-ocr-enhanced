import re

def extract_fields_from_labeled_block(text):
    lines = text.splitlines()
    results = {}

    for i, line in enumerate(lines):
        if line.count(':') >= 2 and i + 1 < len(lines):
            titles = [t.strip() for t in line.split(':') if t.strip()]
            values_line = lines[i + 1].strip()
            values = values_line.split()

            print(f"\n📌 Titles: {titles}")
            print(f"📌 Values: {values}")

            if len(titles) == 2 and len(values) >= 2:
                # First field = everything before the date
                for vi, v in enumerate(values):
                    if re.match(r'\d{2}/\d{2}/\d{4}', v):
                        rep_name = " ".join(values[:vi])
                        order_date = v + " " + " ".join(values[vi+1:])
                        results[titles[0]] = rep_name
                        results[titles[1]] = order_date
                        break

    return results

# ✅ Test input simulating OCR output
text = """
Purchase Representative: Order Deadline:
Haroon Mushtaq 03/05/2025 15:49:41
"""

# Run test
fields = extract_fields_from_labeled_block(text)
print("\n✅ Extracted Fields:")
print("Representative:", fields.get("Purchase Representative"))
print("Order Deadline:", fields.get("Order Deadline"))
