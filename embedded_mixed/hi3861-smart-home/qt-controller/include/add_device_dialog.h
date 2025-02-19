#pragma once

#include <QDialog>

namespace Ui { class AddDeviceDialog; }

class AddDeviceDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AddDeviceDialog(QWidget *parent = nullptr);
    QString getName() const;
    QString getIPAddress() const;
    int getPort() const;
    QString getType() const;

private slots:
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();

private:
    Ui::AddDeviceDialog *ui;
};