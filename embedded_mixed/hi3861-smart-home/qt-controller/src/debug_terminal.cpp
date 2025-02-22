#include "debug_terminal.h"
#include "ui_debug_terminal.h"
#include <QDateTime>

DebugTerminal::DebugTerminal(QTcpSocket* socket, QString& device_name, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DebugTerminal),
    socket(socket)
{
    ui->setupUi(this);
    setWindowTitle(QString("Debug Window (%1)").arg(device_name));
    connect(ui->sendButton, &QPushButton::clicked, this, &DebugTerminal::onSendDataButtonClicked);
}

DebugTerminal::~DebugTerminal() {
    delete ui;
}

void DebugTerminal::onSendDataButtonClicked() {
    QString data = ui->inputLineEdit->text();
    if (!data.isEmpty()) {
        socket->write(data.toUtf8());
        ui->inputLineEdit->clear();
    }
}

void DebugTerminal::addRawData(const QByteArray& byteArray) {
    addMessage(QString::fromUtf8(byteArray));
}

void DebugTerminal::addMessage(const QString& str) {
    ui->outputTextEdit->append(QString("[%1]:").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")));
    ui->outputTextEdit->append(str);
}

void DebugTerminal::closeEvent(QCloseEvent *event) {
    reinterpret_cast<QEvent*>(event)->ignore();
    hide();
}
