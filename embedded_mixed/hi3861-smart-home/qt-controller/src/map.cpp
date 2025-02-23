#include "map.h"

#include <QGraphicsEllipseItem>
#include <QWheelEvent>

Map::Map(QWidget* parent) {
    scene = new QGraphicsScene(parent);
    setScene(scene);

    QPixmap mapPixmap(":/images/map.jpg");
    QGraphicsPixmapItem* mapItem = scene->addPixmap(mapPixmap);

    QGraphicsEllipseItem* marker = new QGraphicsEllipseItem(10,10,10,10);
    marker->setBrush(Qt::red);
    scene->addItem(marker);

    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void Map::wheelEvent(QWheelEvent* event) {
    const double scaleFactor = 1.15;
    if(event->angleDelta().y() > 0)
        scale(scaleFactor,scaleFactor);// 放大
    else
        scale(1.0 / scaleFactor,1.0 / scaleFactor);// 缩小
}