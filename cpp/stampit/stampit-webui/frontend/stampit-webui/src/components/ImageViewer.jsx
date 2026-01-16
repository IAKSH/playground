import { useRef, useState, useEffect } from 'react';
import { Box, Grid, Typography } from '@mui/material';
import BoundaryOverlay from './BoundaryOverlay';

export default function ImageViewer({
  thresholdedUrl,
  finalUrl,
  boundary,
  onSelectionChange,
}) {
  const finalRef = useRef(null);
  const thRef = useRef(null);
  const selRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [startPt, setStartPt] = useState(null);
  const [selRect, setSelRect] = useState(null);

  // 映射选择区域到原始图像坐标
  const getImageSelection = () => {
    const img = finalRef.current;
    if (!img || !selRect) return null;
    const { width: dispW, height: dispH } = img.getBoundingClientRect();
    const scaleX = img.naturalWidth ? img.naturalWidth / dispW : 1;
    const scaleY = img.naturalHeight ? img.naturalHeight / dispH : 1;
    return {
      x: Math.round(selRect.x * scaleX),
      y: Math.round(selRect.y * scaleY),
      w: Math.round(selRect.w * scaleX),
      h: Math.round(selRect.h * scaleY),
    };
  };

  useEffect(() => {
    const mapped = getImageSelection();
    if (mapped) onSelectionChange?.(mapped);
  }, [selRect]);

  const onMouseDown = (e) => {
    const img = finalRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    setDragging(true);
    setStartPt({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setSelRect(null);
  };

  const onMouseMove = (e) => {
    if (!dragging || !startPt) return;
    const img = finalRef.current;
    const rect = img.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const sx = Math.min(startPt.x, x);
    const sy = Math.min(startPt.y, y);
    const sw = Math.abs(x - startPt.x);
    const sh = Math.abs(y - startPt.y);
    setSelRect({ x: sx, y: sy, w: sw, h: sh });
  };

  const onMouseUp = () => setDragging(false);

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle1" gutterBottom>图像预览</Typography>
      <Grid container spacing={2}>
        {/* 阈值图像 */}
        <Grid item xs={12} md={6}>
          <Box className="image-container">
            <img
              ref={thRef}
              src={thresholdedUrl || '/placeholder.png'}
              alt="thresholded"
            />
          </Box>
          <Typography align="center">阈值图像</Typography>
        </Grid>
        {/* 最终图像 */}
        <Grid item xs={12} md={6}>
          <Box className="image-container">
            <img
              ref={finalRef}
              src={finalUrl || '/placeholder.png'}
              alt="final"
              onLoad={() => setSelRect(null)}
            />
            {boundary && (
              <BoundaryOverlay imgRef={finalRef} rect={boundary} />
            )}
            <div
              ref={selRef}
              className="selection-layer"
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              style={{ width: '100%', height: '100%' }}
            />
            {selRect && (
              <div
                style={{
                  position: 'absolute',
                  left: selRect.x,
                  top: selRect.y,
                  width: selRect.w,
                  height: selRect.h,
                  border: '2px solid #1976d2',
                  background: 'rgba(25,118,210,0.1)',
                  pointerEvents: 'none',
                }}
              />
            )}
          </Box>
          <Typography align="center">最终图像</Typography>
        </Grid>
      </Grid>
    </Box>
  );
}
