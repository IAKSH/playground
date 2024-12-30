#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowTitle("TCP Socket Server");
    init_server();
}

MainWindow::~MainWindow()
{
    close_server();
    delete ui;
}

void MainWindow::init_server() {
    server = new QTcpServer(this);

    connect(ui->pushButtonStartup,&QPushButton::clicked,[this](){
        if(server->isListening()) {
            close_server();
            ui->pushButtonStartup->setText("Listen");
            ui->lineEditPort->setEnabled(true);
            ui->lineEditClientId->setEnabled(true);
        }
        else {
            const unsigned short port = ui->lineEditPort->text().toUShort();
            if(server->listen(QHostAddress::Any,port)) {
                ui->pushButtonStartup->setText("Close");
                ui->lineEditPort->setEnabled(false);
                ui->lineEditClientId->setEnabled(false);
            }
        }
    });

    connect(server,&QTcpServer::newConnection,this,[this](){
        while(server->hasPendingConnections()) {
            QTcpSocket *socket = server->nextPendingConnection();
            clients.append(socket);
            ui->textEditRecieve->append(QString("[%1:%2]")
                                        .arg(socket->peerAddress().toString())
                                        .arg(socket->peerPort()));
            connect(socket,&QTcpSocket::readyRead,[this,socket](){
               if(socket->bytesAvailable() <= 0)
                   return;
               const QString recieved_text = QString::fromUtf8(socket->readAll());
               ui->textEditRecieve->append(QString("[%1:%2]")
                                           .arg(socket->peerAddress().toString())
                                           .arg(socket->peerPort()));
               ui->textEditRecieve->append(recieved_text);
            });

#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
            connect(socket, static_cast<void(QAbstractSocket::*)(QAbstractSocket::SocketError)>(&QAbstractSocket::error),
                [this,socket](QAbstractSocket::SocketError){
                ui->textEditRecieve->append(QString("[%1:%2] Soket Error:%3")
                             .arg(socket->peerAddress().toString())
                             .arg(socket->peerPort())
                             .arg(socket->errorString()));
            });
#else
            connect(socket,&QAbstractSocket::errorOccurred,[this,socket](QAbstractSocket::SocketError){
                ui->textEditRecieve->append(QString("[%1:%2] Soket Error:%3")
                             .arg(socket->peerAddress().toString())
                             .arg(socket->peerPort())
                             .arg(socket->errorString()));
            });
#endif
            connect(socket,&QTcpSocket::disconnected,[this,socket]{
                socket->deleteLater();
                clients.removeOne(socket);
                ui->textEditRecieve->append(QString("[%1:%2] Soket Disonnected")
                             .arg(socket->peerAddress().toString())
                             .arg(socket->peerPort()));
            });
        }
        ui->labelClientCount->setText("Client Count: " + clients.size());
    });

    connect(ui->pushButtonSend,&QPushButton::clicked,[this](){
        const unsigned short client_id = ui->lineEditClientId->text().toUShort();
        if(!clients[client_id]->isValid())
            return;
        const QByteArray send_data = ui->textEditSend->toPlainText().toUtf8();
        qDebug() << "sending:" << send_data;
        if(send_data.isEmpty())
            return;
        clients[client_id]->write(send_data);
    });
}

void MainWindow::close_server() {
    server->close();
    for(QTcpSocket* socket : clients) {
        socket->disconnectFromHost();
        if(socket->state() != QAbstractSocket::UnconnectedState)
            socket->abort();
    }
}


