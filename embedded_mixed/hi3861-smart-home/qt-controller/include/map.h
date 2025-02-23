#pragma once

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsObject>
#include <vector>
#include <functional>

class MapMarker : public QGraphicsObject {
    Q_OBJECT

public:
    explicit MapMarker(const QString& name,QObject* parent = nullptr);
    ~MapMarker() = default;
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    void setName(const QString& name);
    const QString& getName() const;

signals:
    void markerMoved(const QPointF& newPos);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    QString name;
    QGraphicsTextItem* textItem;
    QPointF offset;
    bool isDragging;

    void init();
};

class Map : public QGraphicsView {
    Q_OBJECT

using OnMoveCallback = std::function<void(const QPointF&)>; 

public:
    explicit Map(QWidget* parent = nullptr);
   ~Map() = default;
   void addMarker(MapMarker* marker);

protected:
    void wheelEvent(QWheelEvent* event) override;

private:
    QGraphicsScene* scene;
    std::vector<MapMarker*> markers;
};