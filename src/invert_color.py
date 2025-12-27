from PIL import Image

def invert(path: str, name: str):
    img = Image.open(path).convert("RGBA")  # гарантируем RGBA
    img = img.crop((0, 0, img.width - 3, img.height - 3))
    px = img.load()
    new_img = img.copy()
    new_px = new_img.load()

    for h in range(img.height):
        for w in range(img.width):
            if px[w, h] == (255, 255, 255, 255):
                new_px[w, h] = (37, 37, 38, 255)

            elif px[w, h] in [(255, 0, 0, 255), 
                              (0, 0, 255, 255)]: # цвета которые не инвертируются
                continue

            else:
                new_px[w, h] = (255 - px[w, h][0],
                                255 - px[w, h][1],
                                255 - px[w, h][2])

    new_img.save(name)
