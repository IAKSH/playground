#pragma once

#include <QWidget>
#include <QTimer>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>
#include <QTcpSocket>

QT_USE_NAMESPACE

class TemperatureChart : public QWidget
{
    Q_OBJECT

public:
    explicit TemperatureChart(QTcpSocket *socket, const QString &deviceName, QWidget *parent = nullptr);
    ~TemperatureChart() = default;

    void setUpdateFrequency(int milliseconds);
    void addTemperatureData(qreal temperature);

private slots:
    void updateChart();

private:
    QChartView *chartView;
    QChart *chart;
    QLineSeries *series;
    QTimer *timer;
    QTimer *dataTimer; // 确保dataTimer是成员变量
    QString deviceName;
    qreal timeCounter; // 用于模拟时间轴
    QTcpSocket *socket;
};
