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
    explicit TemperatureChart(const QString &deviceName, QWidget *parent = nullptr);
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
    QTimer *dataTimer;
    QString deviceName;
    qreal timeCounter;
};
