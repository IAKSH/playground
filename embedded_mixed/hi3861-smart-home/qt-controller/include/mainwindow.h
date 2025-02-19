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

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow;

class Device {
public:
    QTcpSocket* socket;
    QLabel* connectionStatusLabel;
    QString name;
    DebugTerminal terminal;
    TemperatureChart chart;

    Device(MainWindow* mainWindow,QLabel* connectionStatusLabel,QTcpSocket* socket,QString name);
    ~Device();

private:
    MainWindow* mainWindow;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

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

    void onDebugButtonClicked();
    void onDeleteButtonClicked();
    void onChartButtonClicked();
    void onConnectionStatusLabelClicked(Device* device);
    void onDeviceDiscoveryToggled(bool checked);
    void addDevice(const QString& name, const QString& ipAddress, int port, const QString& type);

    void handleUdpSocket();
};
