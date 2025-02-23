#pragma once

#include <QMainWindow>
#include <QTcpSocket>
#include <QLabel>
#include <QPushButton>
#include <QEvent>
#include <QUdpSocket>
#include <QTableWidgetItem>
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
    QTableWidgetItem* nameTable;
    QString name;
    DebugTerminal terminal;
    TemperatureChart chart;
    MapMarker* marker;
    QPointF pos;

    Device(QTableWidgetItem* nameTable,QLabel* connectionStatusLabel,QTcpSocket* socket,QString name,MainWindow* parent,QWidget* chartWidget);
    ~Device();

    void rename(const QString& newName);

signals:
    void deviceDelete(Device*);
    void deviceRename(Device*);

private slots:
    void onSocketReadyRead();
    void onUpdatePos(const QPointF& newPos);

private:
    MainWindow* mainWindow;
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
    void onDebugButtonClicked();
    void onDeleteButtonClicked();
    void onChartButtonClicked();
    void onRenameButtonClicked();
    void onConnectionStatusLabelClicked(Device* device);
    void onDeviceDiscoveryToggled(bool checked);

private:
    Ui::MainWindow *ui;
    TemperatureChart* showingTempChart;
    std::vector<std::unique_ptr<Device>> devices;
    std::unique_ptr<QUdpSocket> udpSocket;
    
    void addDevice(const QString& name, const QString& ipAddress, int port, const QString& type);
    void handleUdpSocket();
    void setupMap();
    void deleteDevice(Device* device);
    void renameDevice(Device* device);
};
