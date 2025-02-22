#include "temperature_chart.h"
#include <QVBoxLayout>
#include <QValueAxis>
#include <QRegularExpression>

TemperatureChart::TemperatureChart(const QString &deviceName, QWidget *parent)
    : QWidget(parent)
    , deviceName(deviceName)
    , timeCounter(0)
{
    setWindowTitle(QString("Temperature Chart on %1").arg(deviceName));

    // 初始化图表
    chart = new QChart();
    chart->legend()->hide();
    chart->setTitle("Real-time Temperature Data");

    // 初始化折线系列
    series = new QLineSeries();
    chart->addSeries(series);

    // 设置坐标轴
    QValueAxis *axisX = new QValueAxis();
    axisX->setTitleText("Time (s)");
    axisX->setLabelFormat("%.1f");
    axisX->setRange(0, 60); // 显示最近60秒的数据

    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("Temperature (°C)");
    axisY->setRange(0, 100);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);

    series->attachAxis(axisX);
    series->attachAxis(axisY);

    // 创建ChartView
    chartView = new QChartView(chart, this);
    chartView->setRenderHint(QPainter::Antialiasing);

    // 设置尺寸策略为扩展
    chartView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // 布局
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(chartView);

    // 移除边距和间距，使图表填满窗口
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // 设置布局
    this->setLayout(layout);

    // 定时器
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &TemperatureChart::updateChart);
    setUpdateFrequency(1000);

    // 设置窗口属性
    setWindowFlags(Qt::Window);
    resize(800, 600);
}

void TemperatureChart::setUpdateFrequency(int milliseconds)
{
    timer->start(milliseconds);
}

void TemperatureChart::addTemperatureData(qreal temperature)
{
    // 更新时间计数器
    timeCounter += timer->interval() / 1000.0; // 将毫秒转换为秒

    // 添加新数据点
    series->append(timeCounter, temperature);

    // 移除超出范围的数据点，保持显示范围在最近60秒
    while (!series->points().isEmpty() && series->points().first().x() < timeCounter - 60.0) {
        series->remove(0);
    }

    // 更新X轴范围
    QValueAxis *axisX = qobject_cast<QValueAxis *>(chart->axes(Qt::Horizontal).first());
    if (axisX) {
        axisX->setRange(timeCounter - 60.0, timeCounter);
    }
}

void TemperatureChart::updateChart()
{
    // 此处可以添加需要周期性执行的操作
    // 如果没有，可以留空或移除该函数
}
