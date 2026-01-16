import { useEffect, useRef } from 'react';

// Draws boundary rect on a canvas overlay
export default function BoundaryOverlay({ imgRef, rect, color = 'rgba(255,0,0,0.6)', lineWidth = 2 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !rect) return;

    const { width: imgW, height: imgH } = img.getBoundingClientRect();
    canvas.width = imgW;
    canvas.height = imgH;

    const scaleX = imgW / img.naturalWidth;
    const scaleY = imgH / img.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, imgW, imgH);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;

    const x = rect.x * scaleX;
    const y = rect.y * scaleY;
    const w = rect.w * scaleX;
    const h = rect.h * scaleY;

    ctx.strokeRect(x, y, w, h);
  }, [rect]);

  return <canvas ref={canvasRef} className="overlay-canvas" />;
}
