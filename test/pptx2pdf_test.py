import os
from pathlib import Path
from pptxtopdf import convert as pptx_to_pdf_convert

# === Corrected Input and Output Directories ===
input_dir = Path("../local").resolve()
output_dir = Path("../local_output").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Input Dir: {input_dir}")
print(f"📂 Output Dir: {output_dir}")

# === Convert All .pptx Files ===
converted = []
failed = []

for pptx_file in input_dir.glob("*.pptx"):
    try:
        pdf_name = pptx_file.stem + ".pdf"
        output_pdf = output_dir / pdf_name

        if output_pdf.exists():
            print(f"⚠️ Skipped (already exists): {output_pdf.name}")
            continue

        print(f"📊 Converting: {pptx_file.name} → {output_pdf.name}")
        pptx_to_pdf_convert(str(pptx_file.parent), str(output_dir))  # Convert from folder
        converted.append(pptx_file.name)
    except Exception as e:
        print(f"❌ Failed: {pptx_file.name} — {e}")
        failed.append(pptx_file.name)

# === Summary ===
print("\n✅ Conversion complete.")
print(f"✅ Success: {len(converted)} file(s)")
print(f"❌ Failed: {len(failed)} file(s)")
if failed:
    print("  Failed files:", failed)
