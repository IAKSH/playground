#include "debug_terminal.h"
#include "ui_debug_terminal.h"
#include <QDateTime>

DebugTerminal::DebugTerminal(QTcpSocket *socket, const QString& device_name, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DebugTerminal),
    socket(socket)
{
    ui->setupUi(this);
    setWindowTitle(QString("Debug Window (%1)").arg(device_name));
    connect(ui->sendButton, &QPushButton::clicked, this, &DebugTerminal::on_sendButton_clicked);
    connect(socket, &QTcpSocket::readyRead, this, &DebugTerminal::on_readyRead);
}

DebugTerminal::~DebugTerminal()
{
    delete ui;
}

void DebugTerminal::on_sendButton_clicked()
{
    QString data = ui->inputLineEdit->text();
    if (!data.isEmpty()) {
        socket->write(data.toUtf8());
        ui->inputLineEdit->clear();
    }
}

void DebugTerminal::on_readyRead()
{
    QByteArray data = socket->readAll();
    ui->outputTextEdit->append(QString("[%1]:").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    ui->outputTextEdit->append(QString::fromUtf8(data));
}

void DebugTerminal::closeEvent(QCloseEvent *event) {
    reinterpret_cast<QEvent*>(event)->ignore();
    hide();
}
