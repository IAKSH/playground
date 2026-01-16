#include <selectable_label.hpp>

SelectableLabel::SelectableLabel(QWidget *parent)
    : QLabel(parent), selecting(false) {}

QRect SelectableLabel::selectedRect() const {
    return rect.normalized();
}

void SelectableLabel::setSelectionRect(const QRect &r) {
    rect = r.normalized();
    selecting = false;
    update();
    // 立即发出信号
    emit regionSelected(rect);
}

void SelectableLabel::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        selecting = true;
        origin = event->pos();
        rect = QRect(origin, QSize());
        update();
    }
}

void SelectableLabel::mouseMoveEvent(QMouseEvent *event) {
    if (selecting) {
        rect = gen_selected_square(event->pos());
        update();
    }
}

void SelectableLabel::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && selecting) {
        selecting = false;
        rect = gen_selected_square(event->pos());
        update();
        emit regionSelected(rect.normalized());
    }
}

void SelectableLabel::paintEvent(QPaintEvent *event) {
    QLabel::paintEvent(event);
    if (selecting || !rect.isNull()) {
        QPainter painter(this);
        painter.setPen(QPen(Qt::red, 2, Qt::DashLine));
        painter.drawRect(rect.normalized());
    }
}

QRect SelectableLabel::gen_selected_square(QPoint current) {
    int dx = current.x() - origin.x();
    int dy = current.y() - origin.y();
    // 取较大的绝对值，保证宽高一致
    int side = std::max(std::abs(dx), std::abs(dy));
    // 保持拖动方向
    int x = (dx >= 0) ? origin.x() + side : origin.x() - side;
    int y = (dy >= 0) ? origin.y() + side : origin.y() - side;
    return QRect(origin, QPoint(x, y));
}
