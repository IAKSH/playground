#pragma once

#include <QDialog>
#include <QTcpSocket>

namespace Ui {class DebugTerminal;}

class DebugTerminal : public QDialog {
    Q_OBJECT

public:
    explicit DebugTerminal(QTcpSocket *socket, const QString& device_name, QWidget *parent = nullptr);
    ~DebugTerminal();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void on_sendButton_clicked();
    void on_readyRead();

private:
    Ui::DebugTerminal *ui;
    QTcpSocket *socket;
};
