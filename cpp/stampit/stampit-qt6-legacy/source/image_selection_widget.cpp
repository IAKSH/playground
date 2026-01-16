#include "ui_image_selection_widget.h"
#include <image_selection_widget.hpp>
#include <clickable_label.hpp>
#include <selectable_label.hpp>
#include <QFileDialog>
#include <QPixmap>
#include <opencv2/opencv.hpp>

ImageSelectionWidget::ImageSelectionWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::ImageSelectionWidget) {
    ui->setupUi(this);
    connect(ui->load_button,&QPushButton::clicked,this,&ImageSelectionWidget::on_load_button_clicked);
    connect(ui->save_button,&QPushButton::clicked,this,&ImageSelectionWidget::on_save_button_clicked);
    connect(ui->auto_boundary_button, &QPushButton::clicked,this,&ImageSelectionWidget::on_auto_boundary_button_clicked);
    connect(ui->threshold_slider,&QSlider::valueChanged,this,&ImageSelectionWidget::on_threshold_slider_changed);
    connect(ui->threshold_slider, &QSlider::sliderReleased,this,&ImageSelectionWidget::on_threshold_slider_released);
    connect(ui->min_area_label_slider,&QSlider::valueChanged,this,&ImageSelectionWidget::on_min_area_slider_changed);
    connect(ui->min_area_label_slider, &QSlider::sliderReleased,this,&ImageSelectionWidget::on_min_area_slider_released);
    // 如果需要映射到原始图像坐标，可以在这里做比例换算
    //connect(ui->selection_label,&SelectableLabel::regionSelected,this,[=](const QRect &rect){
    //     qDebug() << "selected:" << rect;
    //});

    ui->progress_bar->setVisible(false);
}

ImageSelectionWidget::~ImageSelectionWidget() {
    delete ui;
}

void ImageSelectionWidget::on_load_button_clicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("select image"), "",
        tr("Images (*.png *.jpg *.jpeg *.bmp *.gif)"));

    if (!fileName.isEmpty()) {
        pic = cv::imread(fileName.toStdString());
        if (pic.empty()) {
            qDebug() << "can't open pic!";
        }
        else {
            process_pic();
            apply_thresholded_pic();
            apply_min_boundary();
        }
    }
}

void ImageSelectionWidget::on_save_button_clicked() {
    if (colored.empty()) {
        qDebug() << "No image to save.";
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save cropped image"), "",
        tr("PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"));

    if (fileName.isEmpty()) return;

    QRect selRect = ui->selection_label->selectedRect();
    if (selRect.isNull() || !selRect.isValid()) {
        qDebug() << "No selection rect.";
        return;
    }

    // 映射回原始图像坐标
    QSize labelSize = ui->selection_label->size();
    QSize imgSize(colored.cols, colored.rows);

    double sx = (double)labelSize.width() / imgSize.width();
    double sy = (double)labelSize.height() / imgSize.height();
    double scale = std::min(sx, sy);

    int dispW = (int)std::round(imgSize.width() * scale);
    int dispH = (int)std::round(imgSize.height() * scale);
    int offsetX = (labelSize.width() - dispW) / 2;
    int offsetY = (labelSize.height() - dispH) / 2;

    QRect mapped((selRect.x() - offsetX) / scale,
        (selRect.y() - offsetY) / scale,
        selRect.width() / scale,
        selRect.height() / scale);

    mapped = mapped.intersected(QRect(0, 0, imgSize.width(), imgSize.height()));

    // 用 OpenCV 裁剪
    cv::Rect roi(mapped.x(), mapped.y(), mapped.width(), mapped.height());
    cv::Mat cropped = colored(roi).clone();

    // 确保有 alpha 通道
    if (cropped.channels() == 3) {
        cv::cvtColor(cropped, cropped, cv::COLOR_BGR2BGRA);
    }

    // 将白色像素替换为透明
    for (int y = 0; y < cropped.rows; ++y) {
        cv::Vec4b* line = cropped.ptr<cv::Vec4b>(y);
        for (int x = 0; x < cropped.cols; ++x) {
            cv::Vec4b& px = line[x];
            if (px[0] == 255 && px[1] == 255 && px[2] == 255) {
                px[3] = 0; // alpha 通道设为透明
            }
        }
    }

    // 保存
    std::string outFile = fileName.toStdString();
    if (!cv::imwrite(outFile, cropped)) {
        qDebug() << "Failed to save image.";
    }
}

void ImageSelectionWidget::on_threshold_slider_changed() {
    if (pic.empty()) return;
}

void ImageSelectionWidget::on_threshold_slider_released() {
    if (pic.empty()) return;
    process_pic();
    apply_thresholded_pic();
    apply_min_boundary();
}

void ImageSelectionWidget::on_min_area_slider_changed() {
    if (pic.empty()) return;
}

void ImageSelectionWidget::on_min_area_slider_released() {
    if (pic.empty()) return;
    process_pic();
    apply_thresholded_pic();
    apply_min_boundary();
}

void ImageSelectionWidget::on_auto_boundary_button_clicked() {
    if (pic.empty()) return;
    apply_min_boundary();
}

void ImageSelectionWidget::process_pic() {
    this->setEnabled(false);
    ui->progress_bar->setVisible(true);
    ui->progress_bar->setValue(0);

    int th = ui->threshold_slider->value();
    int min_area = ui->min_area_label_slider->value();
    ui->progress_bar->setValue(10);

    thresholded = threshold(th);
    ui->progress_bar->setValue(40);

    denoised = remove_noise(thresholded.clone(), min_area);
    ui->progress_bar->setValue(70);

    apply_thresholded_pic();
    ui->progress_bar->setValue(90);

    apply_min_boundary();
    ui->progress_bar->setValue(100);

    this->setEnabled(true);
    ui->progress_bar->setVisible(false);
}

void ImageSelectionWidget::apply_thresholded_pic() {
    // 原始灰度图显示在 display_label
    QImage qimgGray(thresholded.data, thresholded.cols, thresholded.rows,
        thresholded.step, QImage::Format_Grayscale8);
    QPixmap pixGray = QPixmap::fromImage(qimgGray.copy());
    ui->display_label->setPixmap(pixGray.scaled(ui->display_label->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation));

    // 用 OpenCV 做颜色替换
    cv::Mat colored(denoised.size(), CV_8UC3, cv::Scalar(255, 255, 255)); // 初始化为白色
    cv::Vec3b replaceColor(0, 0, 255); // BGR 顺序，红色

    for (int y = 0; y < denoised.rows; ++y) {
        const uchar* srcLine = denoised.ptr<uchar>(y);
        cv::Vec3b* dstLine = colored.ptr<cv::Vec3b>(y);
        for (int x = 0; x < denoised.cols; ++x) {
            if (srcLine[x] == 0) {
                dstLine[x] = replaceColor; // 黑色像素 → 替换为红色
            }
        }
    }

    // 保存到成员变量，供后续裁剪/保存使用
    this->colored = colored;

    // 转成 QImage/QPixmap 用于 UI 显示
    QImage qimg_colored(colored.data, colored.cols, colored.rows,
        colored.step, QImage::Format_RGB888);
    QPixmap pix_colored = QPixmap::fromImage(qimg_colored.rgbSwapped());
    // 注意 OpenCV 默认是 BGR，需要 rgbSwapped()

    ui->selection_label->setPixmap(pix_colored.scaled(ui->selection_label->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation));
}

void ImageSelectionWidget::apply_min_boundary() {
    cv::Rect rect = get_min_rect(denoised);

    // 预留像素量，可以在类里定义一个成员变量或常量来控制
    int margin = 10; // 例如预留 10 像素，可根据需要调整

    // 在原始图像坐标系里扩展矩形
    rect.x = std::max(0, rect.x - margin);
    rect.y = std::max(0, rect.y - margin);
    rect.width  = std::min(denoised.cols - rect.x, rect.width  + 2 * margin);
    rect.height = std::min(denoised.rows - rect.y, rect.height + 2 * margin);

    // 获取缩放比例
    double scaleX = static_cast<double>(ui->selection_label->width())  / denoised.cols;
    double scaleY = static_cast<double>(ui->selection_label->height()) / denoised.rows;

    // 映射到 QLabel 坐标
    QRect qrect(rect.x * scaleX,
                rect.y * scaleY,
                rect.width  * scaleX,
                rect.height * scaleY);

    ui->selection_label->setSelectionRect(qrect);
}

cv::Mat ImageSelectionWidget::threshold(int threshold) {
    // 转灰度
    cv::Mat gray;
    cv::cvtColor(pic, gray, cv::COLOR_BGRA2GRAY);
    // 二值化
    cv::Mat bin_ary;
    cv::threshold(gray, bin_ary, threshold, 255, cv::THRESH_BINARY);
    return bin_ary.clone();
}

cv::Rect ImageSelectionWidget::get_min_rect(cv::Mat bin_ary) {
    CV_Assert(bin_ary.type() == CV_8UC1);

    // 原始尺寸
    int origW = bin_ary.cols;
    int origH = bin_ary.rows;

    // 按比例缩放到宽度=256
    int targetW = 256;
    double scale = (double)targetW / origW;
    int targetH = cvRound(origH * scale);

    cv::Mat small;
    cv::resize(bin_ary, small, cv::Size(targetW, targetH), 0, 0, cv::INTER_NEAREST);

    // 在缩小图上做处理（连通域 + minEnclosingCircle）
    cv::Mat inverted;
    cv::bitwise_not(small, inverted);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(inverted, labels, stats, centroids);
    if (nLabels <= 1) return cv::Rect();

    cv::Point2f imgCenter(targetW / 2.0f, targetH / 2.0f);
    double maxArea = 0;
    int bestIdx = -1;

    for (int i = 1; i < nLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 100) continue; // 面积阈值

        cv::Point2f centroid(centroids.at<double>(i,0), centroids.at<double>(i,1));
        double dist = cv::norm(centroid - imgCenter);

        if (dist < std::max(targetW, targetH) * 0.25) {
            if (area > maxArea) {
                maxArea = area;
                bestIdx = i;
            }
        }
    }

    if (bestIdx == -1) return cv::Rect();

    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; ++y) {
        const int* row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == bestIdx) {
                points.emplace_back(x, y);
            }
        }
    }

    if (points.empty()) return cv::Rect();

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(points, center, radius);

    cv::Rect rectSmall(cvRound(center.x - radius),
                       cvRound(center.y - radius),
                       cvRound(radius * 2),
                       cvRound(radius * 2));

    rectSmall &= cv::Rect(0, 0, targetW, targetH);

    // 映射回原始尺寸
    cv::Rect rectOrig(cvRound(rectSmall.x / scale),
                      cvRound(rectSmall.y / scale),
                      cvRound(rectSmall.width / scale),
                      cvRound(rectSmall.height / scale));

    rectOrig &= cv::Rect(0, 0, origW, origH);
    return rectOrig;
}

cv::Mat ImageSelectionWidget::remove_noise(const cv::Mat &binary, int min_area) {
    CV_Assert(binary.type() == CV_8UC1);

    // 反转，把黑色区域当作前景
    cv::Mat inverted;
    cv::bitwise_not(binary, inverted);

    // 连通域分析
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(inverted, labels, stats, centroids);

    cv::Mat cleaned = cv::Mat::zeros(inverted.size(), CV_8UC1);
    for (int i = 1; i < nLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area) {
            cv::Mat mask = (labels == i);
            cleaned.setTo(255, mask);
        }
    }

    // 再反转，恢复原始前景/背景语义
    cv::bitwise_not(cleaned, cleaned);
    return cleaned;
}
