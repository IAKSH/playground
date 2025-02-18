#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QDebug>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowTitle("TCP Socket Client");
    initTcpSocketClient();
}

MainWindow::~MainWindow()
{
    client->abort();
    delete ui;
}

void MainWindow::initTcpSocketClient() {
    client = new QTcpSocket(this);

    connect(ui->pushButtonConnect,&QPushButton::clicked,[this](){
        if(client->state() == QAbstractSocket::ConnectedState) {
            client->abort();
        }
        else if(client->state() == QAbstractSocket::UnconnectedState) {
            const QHostAddress address = QHostAddress(ui->lineEditAddress->text());
            const unsigned short port = ui->lineEditPort->text().toInt();
            client->connectToHost(address,port);
        }
        else {
            qDebug() << "can't connect";
        }
    });

    connect(client,&QTcpSocket::connected,[this](){
        qDebug() << "ban input";
        ui->pushButtonConnect->setEnabled(false);
        ui->lineEditAddress->setEnabled(false);
        ui->lineEditPort->setEnabled(false);
        ui->labelStatus->setText("Connected");
    });

    connect(ui->pushButtonSend,&QPushButton::clicked,[this](){
        if(!client->isValid())
            return;
        const QByteArray send_data = ui->textEditSend->toPlainText().toUtf8();
        qDebug() << "sending:" << send_data;
        if(send_data.isEmpty())
            return;
        client->write(send_data);
    });

    connect(client,&QTcpSocket::readyRead,[this](){
        if(client->bytesAvailable() <= 0)
            return;
        const QString recieved_text = QString::fromUtf8(client->readAll());
        ui->textEditRecieve->append(QString("[%1:%2]").arg(client->peerAddress().toString())
                                    .arg(client->peerPort()));
        ui->textEditRecieve->append(recieved_text);
    });

#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
    connect(client, static_cast<void(QAbstractSocket::*)(QAbstractSocket::SocketError)>(&QAbstractSocket::error),
            [this](QAbstractSocket::SocketError){
        ui->textEditRecieve->append("Socket Error:"+client->errorString());
        QMessageBox msgbox;
        msgbox.setText("Socket Error:" + client->errorString());
        msgbox.exec();
    });
#else
    connect(client,&QAbstractSocket::errorOccurred,[this](QAbstractSocket::SocketError){
        ui->textEditRecieve->append("Socket Error:"+client->errorString());
        QMessageBox msgbox;
        msgbox.setText("Socket Error:" + client->errorString());
        msgbox.exec();
    });
#endif
}
