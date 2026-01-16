#include "zoomable_image_label.hpp"
#include <algorithm>
#include <cmath>

ZoomableImageLabel::ZoomableImageLabel(QWidget *parent)
    : QLabel(parent) {
    setMouseTracking(true);
}

void ZoomableImageLabel::setOriginalPixmap(const QPixmap &pm) {
    originalPixmap = pm;
    update();
}

void ZoomableImageLabel::resizeEvent(QResizeEvent *event) {
    QLabel::resizeEvent(event);
    update();
}

void ZoomableImageLabel::paintEvent(QPaintEvent *event) {
    QLabel::paintEvent(event);

    if (originalPixmap.isNull())
        return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    QSize labelSize = size();
    QSize imgSize = originalPixmap.size();

    double sx = double(labelSize.width())  / imgSize.width();
    double sy = double(labelSize.height()) / imgSize.height();
    currentScale = std::min(sx, sy);

    int disp_w = int(imgSize.width()  * currentScale);
    int disp_h = int(imgSize.height() * currentScale);

    offsetX = (labelSize.width()  - disp_w) / 2;
    offsetY = (labelSize.height() - disp_h) / 2;

    QRect target(offsetX, offsetY, disp_w, disp_h);
    painter.drawPixmap(target, originalPixmap);
}

QPoint ZoomableImageLabel::toImagePos(const QPoint &p) const {
    if (originalPixmap.isNull())
        return p;

    int img_x = (p.x() - offsetX) / currentScale;
    int img_y = (p.y() - offsetY) / currentScale;

    return QPoint(img_x, img_y);
}

QRect ZoomableImageLabel::toWidgetRect(const QRect &r) const {
    if (originalPixmap.isNull())
        return r;

    int x = offsetX + r.x() * currentScale;
    int y = offsetY + r.y() * currentScale;
    int w = r.width()  * currentScale;
    int h = r.height() * currentScale;

    return QRect(x, y, w, h);
}
