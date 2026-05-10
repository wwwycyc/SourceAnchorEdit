export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function readImageDimensions(file) {
  const dataUrl = await fileToDataUrl(file);
  const image = new Image();
  image.src = dataUrl;

  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
  });

  return {
    dataUrl,
    width: image.naturalWidth,
    height: image.naturalHeight,
  };
}

export async function resizeImageToSquare(file, size = 512) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f3f7f1";
  ctx.fillRect(0, 0, size, size);

  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const drawWidth = Math.round(bitmap.width * scale);
  const drawHeight = Math.round(bitmap.height * scale);
  const offsetX = Math.round((size - drawWidth) / 2);
  const offsetY = Math.round((size - drawHeight) / 2);
  ctx.drawImage(bitmap, offsetX, offsetY, drawWidth, drawHeight);
  bitmap.close();

  const dataUrl = canvas.toDataURL("image/png");
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));

  return {
    blob,
    dataUrl,
    base64: dataUrl.split(",")[1],
    resizedWidth: size,
    resizedHeight: size,
  };
}

export async function prepareImageAsset(file, size = 512) {
  const { dataUrl: previewUrl, width, height } = await readImageDimensions(file);
  const resized = await resizeImageToSquare(file, size);

  return {
    file,
    name: file.name,
    previewUrl: resized.dataUrl || previewUrl,
    originalWidth: width,
    originalHeight: height,
    resizedWidth: resized.resizedWidth,
    resizedHeight: resized.resizedHeight,
    base64: resized.base64,
    blob: resized.blob,
    mimeType: "image/png",
  };
}
