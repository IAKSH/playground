#include "ui_image_selection_widget.h"
#include <image_selection_widget.hpp>
#include <clickable_label.hpp>
#include <selectable_image_label.hpp>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>

ImageSelectionWidget::ImageSelectionWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::ImageSelectionWidget) {
    ui->setupUi(this);
    connect(ui->load_button,&QPushButton::clicked,this,&ImageSelectionWidget::on_load_button_clicked);
    connect(ui->save_button,&QPushButton::clicked,this,&ImageSelectionWidget::on_save_button_clicked);
    connect(ui->threshold_slider,&QSlider::valueChanged,this,&ImageSelectionWidget::on_threshold_slider_changed);
    connect(ui->threshold_slider, &QSlider::sliderReleased,this,&ImageSelectionWidget::on_threshold_slider_released);
    connect(ui->min_area_label_slider,&QSlider::valueChanged,this,&ImageSelectionWidget::on_min_area_slider_changed);
    connect(ui->min_area_label_slider, &QSlider::sliderReleased,this,&ImageSelectionWidget::on_min_area_slider_released);
    // 如果需要映射到原始图像坐标，可以在这里做比例换算
    //connect(ui->selection_label,&SelectableLabel::regionSelected,this,[=](const QRect &rect){
    //     qDebug() << "selected:" << rect;
    //});

    //ui->progress_bar->setVisible(false);
}

ImageSelectionWidget::~ImageSelectionWidget() {
    delete ui;
}

void ImageSelectionWidget::on_load_button_clicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("select image"), "",
        tr("Images (*.png *.jpg *.jpeg *.bmp *.gif)"));

    if (!fileName.isEmpty()) {
        original_img.load(fileName.toStdString());
        if (original_img.empty()) {
            qDebug() << "can't open pic!";
        }
        else {
            auto img = original_img.copy();
            process_img(img);
        }
    }
}

void ImageSelectionWidget::on_save_button_clicked() {
    if (final_img.empty()) {
        qDebug() << "No image to save.";
        return;
    }

    QString file_name = QFileDialog::getSaveFileName(this,
        tr("Save cropped image"), "",
        tr("PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"));

    if (file_name.isEmpty()) return;
    if (!file_name.contains('.')) { 
        file_name += ".png";
    }

    QRect sel_rect = ui->selection_label->selectedRect();
    qDebug() << "selected: " << sel_rect;
    if (sel_rect.isNull() || !sel_rect.isValid()) {
        qDebug() << "No selection rect.";
        return;
    }

    QRect mapped = sel_rect.intersected(QRect(0,0,final_img.width(),final_img.height()));
    qDebug() << "save final img with mapped rect:" << mapped;
    final_img.save_with_transparent_pix(file_name.toStdString(),
                   mapped.x(), mapped.y(),
                   mapped.width(), mapped.height());
}

void ImageSelectionWidget::on_threshold_slider_changed() {
    if (original_img.empty()) return;
}

void ImageSelectionWidget::on_threshold_slider_released() {
    if (original_img.empty()) return;
    auto img = original_img.copy();
    process_img(img);
}

void ImageSelectionWidget::on_min_area_slider_changed() {
    if (original_img.empty()) return;
}

void ImageSelectionWidget::on_min_area_slider_released() {
    if (original_img.empty()) return;
    auto img = original_img.copy();
    process_img(img);
}

void ImageSelectionWidget::process_img(stampit::Image& img) {
    this->setEnabled(false);
    //ui->progress_bar->setVisible(true);
    ui->progress_bar->setValue(0);

    int th = ui->threshold_slider->value();
    int min_area = ui->min_area_label_slider->value();
    ui->progress_bar->setValue(10);

    img.threshold(th);
    ui->progress_bar->setValue(20);
    apply_thresholded_img(img);
    ui->progress_bar->setValue(25);

    img.denoise(min_area);
    ui->progress_bar->setValue(50);

    auto rect = img.get_min_boundary(10);
    ui->progress_bar->setValue(70);
    apply_auto_boundary(rect);
    ui->progress_bar->setValue(80);

    img.set_color(255,0,0);
    ui->progress_bar->setValue(90);
    apply_final_img(img);
    ui->progress_bar->setValue(100);

    this->setEnabled(true);
    //ui->progress_bar->setVisible(false);

    final_img = img;
}

void ImageSelectionWidget::apply_thresholded_img(stampit::Image& img) {
    QImage qimg(img.data(), img.width(), img.height(),
        img.__mat.step, QImage::Format_Grayscale8);
    QPixmap qpix = QPixmap::fromImage(qimg.copy());
    ui->display_label->setOriginalPixmap(qpix);
}

void ImageSelectionWidget::apply_final_img(stampit::Image& img) {
    QImage qimg(img.data(), img.width(), img.height(),
        img.__mat.step, QImage::Format_RGB888);
    // OpenCV 默认是 BGR，需要 rgbSwapped()
    QPixmap qpix = QPixmap::fromImage(qimg.rgbSwapped());
    ui->selection_label->setOriginalPixmap(qpix);
}


void ImageSelectionWidget::apply_auto_boundary(stampit::Rect& rect) {
    QRect imageRect(rect.x, rect.y, rect.w, rect.h);
    qDebug() << "auto selected: " << imageRect;
    ui->selection_label->setSelectionRect(imageRect);
}
