#pragma once

#include <QGraphicsScene>
#include <QGraphicsView>

class Map : public QGraphicsView {
    Q_OBJECT;

public:
    explicit Map(QWidget* parent = nullptr);
   ~Map() = default;

protected:
    void wheelEvent(QWheelEvent* event) override;

private:
    QGraphicsScene* scene;
};