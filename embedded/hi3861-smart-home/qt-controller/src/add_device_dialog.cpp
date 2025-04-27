#include "add_device_dialog.h"
#include "ui_add_device_dialog.h"
#include <QDialog>
#include <QDialogButtonBox>

AddDeviceDialog::AddDeviceDialog(QWidget *parent) :
    QDialog(parent), ui(new Ui::AddDeviceDialog) {
    ui->setupUi(this);
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &AddDeviceDialog::on_buttonBox_accepted);
    connect(ui->buttonBox, &QDialogButtonBox::rejected, this,&AddDeviceDialog::on_buttonBox_rejected);
}

QString AddDeviceDialog::getName() const {
    return ui->lineEditName->text();
}

QString AddDeviceDialog::getIPAddress() const {
    return ui->lineEditIPAddress->text();
}

int AddDeviceDialog::getPort() const {
    return ui->spinBoxPort->value();
}

QString AddDeviceDialog::getType() const {
    return ui->comboBoxType->currentText();
}

void AddDeviceDialog::on_buttonBox_accepted() {
    accept();
}

void AddDeviceDialog::on_buttonBox_rejected() {
    reject();
}