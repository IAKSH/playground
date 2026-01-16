#include "selectable_image_label.hpp"
#include <algorithm>

SelectableImageLabel::SelectableImageLabel(QWidget *parent)
    : ZoomableImageLabel(parent) {
    setMouseTracking(true);
}

void SelectableImageLabel::setSelectionRect(const QRect &r) {
    imageRect = r.normalized();
    selecting = false;
    update();
    emit regionSelected(imageRect);
}

void SelectableImageLabel::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && !originalPixmap.isNull()) {
        selecting = true;
        origin = toImagePos(event->pos());
        imageRect = QRect(origin, QSize());
        update();
    }
}

void SelectableImageLabel::mouseMoveEvent(QMouseEvent *event) {
    if (selecting && !originalPixmap.isNull()) {
        QPoint imgPos = toImagePos(event->pos());
        imageRect = gen_selected_square(imgPos);
        update();
    }
}

void SelectableImageLabel::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && selecting && !originalPixmap.isNull()) {
        selecting = false;
        QPoint imgPos = toImagePos(event->pos());
        imageRect = gen_selected_square(imgPos).normalized();
        update();
        emit regionSelected(imageRect);
    }
}

void SelectableImageLabel::paintEvent(QPaintEvent *event) {
    ZoomableImageLabel::paintEvent(event);

    if (imageRect.isNull())
        return;

    QPainter painter(this);
    painter.setPen(QPen(Qt::red, 2, Qt::DashLine));

    QRect widgetRect = toWidgetRect(imageRect.normalized());
    painter.drawRect(widgetRect);
}

QRect SelectableImageLabel::gen_selected_square(const QPoint &current) {
    int dx = current.x() - origin.x();
    int dy = current.y() - origin.y();

    int side = std::max(std::abs(dx), std::abs(dy));

    int x = (dx >= 0) ? origin.x() + side : origin.x() - side;
    int y = (dy >= 0) ? origin.y() + side : origin.y() - side;

    return QRect(origin, QPoint(x, y));
}
