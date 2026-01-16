#pragma once
#include <QLabel>
#include <QPainter>

class ZoomableImageLabel : public QLabel {
    Q_OBJECT
public:
    explicit ZoomableImageLabel(QWidget *parent = nullptr);

    void setOriginalPixmap(const QPixmap &pm);
    QSize imageSize() const { return originalPixmap.size(); }

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

    // 坐标转换（供子类使用）
    QPoint toImagePos(const QPoint &p) const;
    QRect  toWidgetRect(const QRect &r) const;

protected:
    QPixmap originalPixmap;

    // 当前缩放信息（每次 paintEvent 自动计算）
    double currentScale = 1.0;
    int offsetX = 0;
    int offsetY = 0;
};
