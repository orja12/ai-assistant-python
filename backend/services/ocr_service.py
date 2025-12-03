import pytesseract
from PIL import Image, ImageFilter, ImageOps

def enhance_image(img_path):
    # تحويل لصورة grayscale
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)

    # زيادة الوضوح
    img = img.filter(ImageFilter.SHARPEN)

    # حفظ الصورة المحسّنة
    enhanced = img_path + "_enhanced.png"
    img.save(enhanced)
    return enhanced

def extract_text(img_path, lang="ara+eng"):
    # تحسين الصورة قبل استخراج النص
    enhanced = enhance_image(img_path)

    # استخدام Tesseract لاستخراج النص
    text = pytesseract.image_to_string(Image.open(enhanced), lang=lang)

    # تنظيف النص من المسافات الزائدة
    return text.strip()
