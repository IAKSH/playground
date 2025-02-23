#include "map.h"

#include <QGraphicsEllipseItem>
#include <QWheelEvent>
#include <QGraphicsSceneMouseEvent>
#include <QMenu>

Map::Map(QWidget* parent) {
    scene = new QGraphicsScene(parent);
    setScene(scene);

    QPixmap mapPixmap(":/images/map.jpg");
    QGraphicsPixmapItem* mapItem = scene->addPixmap(mapPixmap);

    //MapMarker* marker = new MapMarker("nihao");
    //scene->addItem(marker);

    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void Map::addMarker(MapMarker* marker) {
    scene->addItem(marker);
    markers.emplace_back(marker);
}

void Map::wheelEvent(QWheelEvent* event) {
    const double scaleFactor = 1.15;
    if(event->angleDelta().y() > 0)
        scale(scaleFactor,scaleFactor);// 放大
    else
        scale(1.0 / scaleFactor,1.0 / scaleFactor);// 缩小
}

MapMarker::MapMarker(const QString& name,QObject* parent) 
    : name(name),isDragging(false)
{
    init();
}

void MapMarker::setName(const QString& name) {
    this->name = name;
    textItem->setPlainText(name);
}

const QString &MapMarker::getName() const {
    return name;
}

QRectF MapMarker::boundingRect() const {
    // 标记点的Rect
    return QRectF(-10,-10,20,40);
}

void MapMarker::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setBrush(Qt::red);
    painter->setPen(Qt::NoPen);
    painter->drawEllipse(-5,-5,10,10);
}

void MapMarker::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    if(event->button() == Qt::LeftButton) {
        isDragging = true;
        offset = event->pos();
        event->accept();
    }
    else
        event->ignore();
}

void MapMarker::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    if(isDragging) {
        QPointF newPos = mapToParent(event->pos() - offset);
        setPos(newPos);
        event->accept();
    }
    else
        event->ignore();
}

void MapMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    if(isDragging) {
        isDragging = false;
        emit markerMoved(this->pos());
        event->accept();
    }
    else
        event->ignore();
}

void MapMarker::contextMenuEvent(QGraphicsSceneContextMenuEvent* event) {
    QMenu menu;
    QAction* deleteAction = menu.addAction("Delete");
    QAction* renameAction = menu.addAction("Rename");
    QAction* shiftAction = menu.addAction("Shift");
    QAction* selectedAction = menu.exec(event->screenPos());

    if(selectedAction == deleteAction) {
        emit markerDelete();
        event->accept();
    }
    else if(selectedAction == renameAction) {
        emit markerRename();
        event->accept();
    }
    event->ignore();
}

void MapMarker::init() {
    // 设置该QGraphicsObject为可选择，可拖动
    setFlags(ItemIsSelectable | ItemIsMovable);

    // 创建名称文本项
    textItem = new QGraphicsTextItem(name,this);
    textItem->setDefaultTextColor(Qt::black);
    textItem->setFont(QFont("Arial",12));
    // 设置位置为标记点上方
    textItem->setPos(-textItem->boundingRect().width() / 2,-30);
}
