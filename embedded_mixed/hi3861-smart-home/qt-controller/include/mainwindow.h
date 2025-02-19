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

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow;

class Device {
public:
    QTcpSocket* socket;
    QLabel* connectionStatusLabel;
    DebugTerminal terminal;
    QString name;

    Device(MainWindow* main_window,QLabel* connectionStatusLabel,QTcpSocket* socket,QString name);
    ~Device();

private:
    MainWindow* main_window;
};

class MainWindow : public QMainWindow
{
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
    bool enableAutoDiscovery = true;

    void onDebugButtonClicked();
    void onDeleteButtonClicked();
    void onConnectionStatusLabelClicked(Device* device);
    void addDevice(const QString& name, const QString& ipAddress, int port, const QString& type);
};
