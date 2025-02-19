#include "mainwindow.h"
#include "ui_MainWindow.h"
#include "add_device_dialog.h"
#include <QTcpSocket>
#include <QMessageBox>
#include <algorithm>
#include <QStringList>

#define DEVICE_UDP_BROADCAST_PORT 12345

Device::Device(MainWindow* main_window,QLabel* connection_status_label,QTcpSocket* socket,QString name) 
    : main_window(main_window), connectionStatusLabel(connection_status_label),
        socket(socket), terminal(socket, name, main_window), name(name) {}

Device::~Device() {
    // 虽然socket和connectionStatusLabel在创建的时候就绑定了主窗口为父组件
    // 但这只能保证关闭主窗口时，这俩会自动销毁，但是并不能在只删除设备时销毁
    // 所以这里手动将其随设备一起销毁

    // 断开socket连接并删除
    if (socket) {
        socket->disconnect();
        socket->abort();
        socket->deleteLater();
    }
    // 删除ui上的状态标签
    if (connectionStatusLabel) {
        connectionStatusLabel->removeEventFilter(main_window);
        connectionStatusLabel->deleteLater();
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    connect(ui->addDeviceButton, &QPushButton::clicked,
            this, &MainWindow::onAddDeviceButtonClicked);
    connect(ui->deviceDiscoveryToggle,&QCheckBox::toggled,this,onDeviceDiscoveryToggled);

    handleUdpSocket();
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::onAddDeviceButtonClicked() {
    AddDeviceDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted) {
        QString name = dialog.getName();
        QString ipAddress = dialog.getIPAddress();
        int port = dialog.getPort();
        QString type = dialog.getType();

        addDevice(name, ipAddress, port, type);
    }
}

void MainWindow::addDevice(const QString& name, const QString& ipAddress, int port, const QString& type) {
    int row = ui->deviceTable->rowCount();
    ui->deviceTable->insertRow(row);

    auto imageLabel = new QLabel(this);
    imageLabel->setPixmap(QPixmap("images/default_device.png").scaled(50, 50, Qt::KeepAspectRatio));
    ui->deviceTable->setCellWidget(row, 0, imageLabel);

    ui->deviceTable->setItem(row, 1, new QTableWidgetItem(name));
    ui->deviceTable->setItem(row, 2, new QTableWidgetItem(ipAddress));
    ui->deviceTable->setItem(row, 3, new QTableWidgetItem(QString::number(port)));
    ui->deviceTable->setItem(row, 4, new QTableWidgetItem(type));

    auto connectionStatusLabel = new QLabel("Connecting", this);
    connectionStatusLabel->setStyleSheet("QLabel { color : black; }");
    ui->deviceTable->setCellWidget(row, 5, connectionStatusLabel);

    auto debugButton = new QPushButton("Debug", this);
    ui->deviceTable->setCellWidget(row, 6, debugButton);
    connect(debugButton, &QPushButton::clicked, this, onDebugButtonClicked);

    auto deleteButton = new QPushButton("Delete", this);
    ui->deviceTable->setCellWidget(row, 7, deleteButton);
    connect(deleteButton, &QPushButton::clicked, this, onDeleteButtonClicked);

    auto socket = new QTcpSocket(this);
    
    devices.emplace_back(std::make_unique<Device>(this,connectionStatusLabel,socket,name));
    Device* device_ptr = devices.back().get(); // 获取原始指针

    connect(socket, &QTcpSocket::connected, this, [this,device_ptr](){updateConnectionStatus(device_ptr);});
    connect(socket, &QTcpSocket::disconnected, this, [this,device_ptr](){updateConnectionStatus(device_ptr);});
    connect(socket, &QAbstractSocket::errorOccurred, this, [this,device_ptr](){updateConnectionStatus(device_ptr);});
    socket->connectToHost(ipAddress, port);

    // 在按钮上设置设备属性
    debugButton->setProperty("device", QVariant::fromValue(device_ptr));
    deleteButton->setProperty("device", QVariant::fromValue(device_ptr));

    socket->setProperty("device", QVariant::fromValue(device_ptr));

    connectionStatusLabel->setProperty("device", QVariant::fromValue(device_ptr));
    connectionStatusLabel->installEventFilter(this);
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::MouseButtonRelease) {
        QLabel* label = qobject_cast<QLabel*>(obj);
        if(label) {
            Device* device = label->property("device").value<Device*>();
            if(device) {
                onConnectionStatusLabelClicked(device);
                return true;
            }
        }
    }

    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::onDebugButtonClicked() {
    QPushButton* button = qobject_cast<QPushButton*>(sender());
    if (!button) return;
    Device* device = button->property("device").value<Device*>();
    if (!device) return;

    device->terminal.show();
}

void MainWindow::onConnectionStatusLabelClicked(Device* device) {
    if(!device) return;

    QTcpSocket* socket = device->socket;

    if (socket->state() != QAbstractSocket::ConnectedState &&
        socket->state() != QAbstractSocket::ConnectingState)
    {
        auto deviceName = device->name;
        auto errorString = socket->errorString();
        auto reply = QMessageBox::question(this, "Connection Failed",
            QString("Failed to connect to device: %1\nError: %2\n"
                    "Would you like to reconnect?")
            .arg(deviceName).arg(errorString),
            QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            QString ipAddress = socket->peerName();
            int port = socket->peerPort();

            // 清除之前的错误状态
            socket->abort();

            // 尝试重连
            socket->connectToHost(ipAddress,port);
            // 等待3s连接
            if(!socket->waitForConnected(3000)) {
                QMessageBox::warning(this,"Connection failed!",QString("can't connect to device %1\nerror: %2")
                .arg(deviceName)
                .arg(socket->errorString()));
            }
        }
    }
}

void MainWindow::updateConnectionStatus(Device* device) {
    if(!device) return;

    QLabel* connection_status_label = device->connectionStatusLabel;
    QTcpSocket* socket = device->socket;

    switch (socket->state()) {
    case QAbstractSocket::ConnectedState:
        connection_status_label->setText("Connected");
        connection_status_label->setStyleSheet("QLabel { color : green; }");
        break;
    case QAbstractSocket::ConnectingState:
        connection_status_label->setText("Connecting");
        connection_status_label->setStyleSheet("QLabel { color : black; }");
        break;
    default:
        connection_status_label->setText("No Connection");
        connection_status_label->setStyleSheet("QLabel { color : red; }");
        break;
    }
}

void MainWindow::onDeleteButtonClicked() {
    QPushButton* button = qobject_cast<QPushButton*>(sender());
    if (!button) return;
    Device* device = button->property("device").value<Device*>();
    if (!device) return;

    // 在devices向量中找到设备并移除
    auto it = std::find_if(devices.begin(), devices.end(),[device](std::unique_ptr<Device>& dev){return dev.get() == device;});
    if (it != devices.end()) {
        // 从表格中移除对应的行
        int row = ui->deviceTable->indexAt(button->pos()).row();
        ui->deviceTable->removeRow(row);
        // 从devices向量中移除设备
        devices.erase(it);
    }
}

void MainWindow::processPendingDatagrams() {
    while (udpSocket->hasPendingDatagrams()) {
        QByteArray datagram;
        QHostAddress sender;
        quint16 senderPort;
        
        datagram.resize(udpSocket->pendingDatagramSize());
        udpSocket->readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);

        QString data(datagram);
        QStringList parts = data.split(',');

        if (parts.size() >= 3) { // 更新判断条件
            QString name = parts[0].mid(parts[0].indexOf(":") + 2);
            QString type = parts[1].mid(parts[1].indexOf(":") + 2);
            int port = parts[2].mid(parts[2].indexOf(":") + 2).toInt();

            // 从sender获取设备地址
            QString address = sender.toString();

            // 检查设备是否已经存在
            bool deviceExists = false;
            for (int i = 0; i < ui->deviceTable->rowCount(); ++i) {
                QString existingName = ui->deviceTable->item(i, 1)->text();
                QString existingType = ui->deviceTable->item(i, 4)->text();
                QString existingAddress = ui->deviceTable->item(i, 2)->text();
                int existingPort = ui->deviceTable->item(i, 3)->text().toInt();

                if (name == existingName && type == existingType && address == existingAddress && port == existingPort) {
                    deviceExists = true;
                    break;
                }
            }

            if (!deviceExists) {
                addDevice(name, address, port, type);
            }
        }
    }
}

void MainWindow::onDeviceDiscoveryToggled(bool checked) {
    handleUdpSocket();
}

void MainWindow::handleUdpSocket() {
    if(udpSocket) {
        udpSocket.reset();
    }
    else {
        udpSocket = std::make_unique<QUdpSocket>(this);
        udpSocket->bind(QHostAddress::Any, DEVICE_UDP_BROADCAST_PORT, QUdpSocket::ShareAddress);
        connect(udpSocket.get(), &QUdpSocket::readyRead,
            this, &MainWindow::processPendingDatagrams);
    }
}