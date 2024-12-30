#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QDebug>
#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowTitle("A Qt Home Control");
    init_client();
    init_chart();

    // Initialize the buffer for storing temperature readings
    for(int i = 0; i < 30; ++i) {
        tempBuffer.append(0);
    }

    // Create and start a timer to update the chart every second
    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::update_chart);
    timer->start(1000);
}

MainWindow::~MainWindow()
{
    client->abort();
    delete ui;
}

void MainWindow::init_client() {
    client = new QTcpSocket(this);

    connect(ui->pushButtonConnect,&QPushButton::clicked,[this](){
        if(client->state() == QAbstractSocket::ConnectedState) {
            client->abort();
        }
        else if(client->state() == QAbstractSocket::UnconnectedState) {
            const QHostAddress address = QHostAddress("192.168.2.123");
            //const QHostAddress address = QHostAddress("127.0.0.1");
            const unsigned short port = 8888;
            client->connectToHost(address,port);
        }
        else {
            qDebug() << "can't connect";
        }
    });

    connect(client,&QTcpSocket::connected,[this](){
        qDebug() << "connected";
        ui->labelConnectionStatus->setText("Connected");
        ui->pushButtonConnect->setEnabled(false);
    });

    connect(client,&QTcpSocket::readyRead,[this](){
        if(client->bytesAvailable() <= 0)
            return;
        const QString received_text = QString::fromUtf8(client->readAll());
        qDebug() << "received:" << received_text;

        float temp = received_text.toFloat();
        ui->labelTemp->setText(QString("Temp: %1").arg(temp));
        ui->labelOverHeated->setText(QString("Over Heated: %1").arg(temp >= 40.0f ? "true" : "false"));

        // Add the new temperature reading to the buffer
        tempBuffer.append(temp);
        if (tempBuffer.size() > 30) {
            tempBuffer.removeFirst();
        }
    });

    connect(ui->checkBoxLed0,&QCheckBox::stateChanged,[this](){
        bool b = ui->checkBoxLed0->isChecked();
        const int led_id = 0;
        qDebug() << "led" << led_id << ":" << b;
        client->write(QString("led%1 %2").arg(led_id).arg(b ? "on" : "off").toUtf8());
    });

    connect(ui->checkBoxLed1,&QCheckBox::stateChanged,[this](){
        bool b = ui->checkBoxLed1->isChecked();
        const int led_id = 1;
        qDebug() << "led" << led_id << ":" << b;
        client->write(QString("led%1 %2").arg(led_id).arg(b ? "on" : "off").toUtf8());
    });

    connect(ui->checkBoxLed2,&QCheckBox::stateChanged,[this](){
        bool b = ui->checkBoxLed2->isChecked();
        const int led_id = 2;
        qDebug() << "led" << led_id << ":" << b;
        client->write(QString("led%1 %2").arg(led_id).arg(b ? "on" : "off").toUtf8());
    });

    connect(ui->checkBoxLed3,&QCheckBox::stateChanged,[this](){
        bool b = ui->checkBoxLed3->isChecked();
        const int led_id = 3;
        qDebug() << "led" << led_id << ":" << b;
        client->write(QString("led%1 %2").arg(led_id).arg(b ? "on" : "off").toUtf8());
    });

#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
    connect(client, static_cast<void(QAbstractSocket::*)(QAbstractSocket::SocketError)>(&QAbstractSocket::error),
            [this](QAbstractSocket::SocketError){
        QMessageBox msgbox;
        msgbox.setText("Socket Error:" + client->errorString());
        msgbox.exec();

        ui->labelConnectionStatus->setText("No Connection");
        ui->pushButtonConnect->setEnabled(true);
   });
#else
   connect(client,&QAbstractSocket::errorOccurred,[this](QAbstractSocket::SocketError){
       QMessageBox msgbox;
       msgbox.setText("Socket Error:" + client->errorString());
       msgbox.exec();

       ui->labelConnectionStatus->setText("No Connection");
       ui->pushButtonConnect->setEnabled(true);
   });
#endif
}

void MainWindow::init_chart() {
    QChart* chart = new QChart();
    chart->setTitle("Temperature over Time");
    ui->graphicsView->setChart(chart);
    ui->graphicsView->setRenderHint(QPainter::Antialiasing);
    ui->graphicsView->chart()->setTheme(QChart::ChartTheme(0));

    series = new QLineSeries();
    series->setName("Temperature");
    chart->addSeries(series);

    series->setPointsVisible(false);

    axis_x = new QValueAxis;
    axis_x->setRange(0, 30);
    axis_x->setLabelFormat("%d s");
    axis_x->setTickCount(6);
    axis_x->setMinorTickCount(1);

    axis_y = new QValueAxis;
    axis_y->setRange(0, 100);

    chart->setAxisX(axis_x, series);
    chart->setAxisY(axis_y, series);

    series->clear();
}

void MainWindow::update_chart() {
    series->clear();
    for(int i = 0; i < tempBuffer.size(); ++i) {
        series->append(i, tempBuffer.at(i));
    }
}
