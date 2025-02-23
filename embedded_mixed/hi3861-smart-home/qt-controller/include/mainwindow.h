#pragma once

#include <QMainWindow>
#include <QTcpSocket>
#include <QLabel>
#include <QPushButton>
#include <QEvent>
#include <QUdpSocket>
#include <vector>
#include <map>

#include "debug_terminal.h"
#include "temperature_chart.h"
#include "map.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow;

class Device : public QObject {
    Q_OBJECT

public:
    QTcpSocket* socket;
    QLabel* connectionStatusLabel;
    QString name;
    DebugTerminal terminal;
    TemperatureChart chart;
    MapMarker* marker;
    QPointF pos;

    Device(QLabel* connectionStatusLabel,QTcpSocket* socket,QString name,MainWindow* parent,QWidget* chartWidget);
    ~Device();

private slots:
    void onSocketReadyRead();

private:
    MainWindow* mainWindow;

    void onUpdatePos(const QPointF& newPos);
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    Map* map;

    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void setShowingTempChart(TemperatureChart& chart);

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void onAddDeviceButtonClicked();
    void updateConnectionStatus(Device* device);
    void processPendingDatagrams();

private:
    Ui::MainWindow *ui;

    std::vector<std::unique_ptr<Device>> devices;
    std::unique_ptr<QUdpSocket> udpSocket;

    TemperatureChart* showingTempChart;

    void onDebugButtonClicked();
    void onDeleteButtonClicked();
    void onChartButtonClicked();
    void onConnectionStatusLabelClicked(Device* device);
    void onDeviceDiscoveryToggled(bool checked);
    void addDevice(const QString& name, const QString& ipAddress, int port, const QString& type);

    void handleUdpSocket();

    void setupMap();
};
