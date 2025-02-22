#pragma once

#include <QDialog>
#include <QTcpSocket>

namespace Ui {class DebugTerminal;}

class DebugTerminal : public QDialog {
    Q_OBJECT

public:
    explicit DebugTerminal(QTcpSocket* socket, QString& device_name, QWidget *parent = nullptr);
    ~DebugTerminal();

    void addRawData(const QByteArray& byteArray);
    void addMessage(const QString& str);

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onSendDataButtonClicked();

private:
    Ui::DebugTerminal *ui;
    QTcpSocket* socket;
};
