#pragma once
#include "zoomable_image_label.hpp"
#include <QMouseEvent>

class SelectableImageLabel : public ZoomableImageLabel {
    Q_OBJECT
public:
    explicit SelectableImageLabel(QWidget *parent = nullptr);

    QRect selectedRect() const { return imageRect; }
    void setSelectionRect(const QRect &r);

signals:
    void regionSelected(const QRect &rect);

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    QRect gen_selected_square(const QPoint &current);

private:
    bool selecting = false;
    QPoint origin;
    QRect imageRect;
};
