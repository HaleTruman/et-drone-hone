(function registerColorMaskRenderer(global) {
  const Shared = global.VisionReview.shared;

  function render(context, panel) {
    const { ctx, width, height } = Shared.prepareCanvas(context, panel);
    if (!context.maskBuffer) {
      Shared.showNote(panel, context.maskError || "No color-mask binary was written for this frame.");
      return;
    }

    const mask = new Uint8Array(context.maskBuffer);
    const expectedLength = width * height;
    if (mask.length < expectedLength) {
      Shared.showNote(panel, `Mask binary is too small: ${mask.length} bytes for ${expectedLength} pixels.`);
      return;
    }

    const image = ctx.createImageData(width, height);
    for (let index = 0; index < expectedLength; index += 1) {
      const bits = mask[index] || 0;
      const offset = index * 4;
      if (bits === 0) {
        image.data[offset + 3] = 0;
        continue;
      }
      const bitIndex = Math.max(0, Math.floor(Math.log2(bits & -bits)));
      const [red, green, blue] = Shared.rgbFromHex(Shared.colorForBit(context, bitIndex));
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 145;
    }
    ctx.putImageData(image, 0, 0);
    Shared.hideNote(panel);
  }

  global.VisionStageRenderers.color_masking = { render };
})(window);
