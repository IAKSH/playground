#pragma once
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>

class SelectableLabel : public QLabel {
    Q_OBJECT
public:
    explicit SelectableLabel(QWidget *parent = nullptr);
    QRect selectedRect() const;
    void setSelectionRect(const QRect &r);

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

signals:
    void regionSelected(const QRect &rect);

private:
    bool selecting;
    QPoint origin;
    QRect rect;
    QRect gen_selected_square(QPoint current);
};
