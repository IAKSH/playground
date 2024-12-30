#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTcpServer>
#include <QTcpSocket>
#include <QtCharts>
QT_CHARTS_USE_NAMESPACE

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QTcpSocket* client;
    QLineSeries* series;
    QValueAxis* axis_x;
    QValueAxis* axis_y;
    QList<float> tempBuffer;

    void init_client();
    void init_chart();
    void update_chart();

};
#endif // MAINWINDOW_H
