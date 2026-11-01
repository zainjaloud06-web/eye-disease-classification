import os
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from config import MODEL_DIR, Patch_size, patience
from preprocessing1 import create_data_generators
from visualization import plot_training_history  

def fine_tune_model(
    start_checkpoint='best_model.keras',
    checkpoint_name='trainig_checkpoint.keras',
    new_epochs=10,
    new_lr=1e-5,
    resume=True
):
    """
    يقوم بعمل Fine-Tuning للنموذج المدرب مسبقاً مع إمكانية استئناف التدريب من آخر Epoch.
    - لا يغير ملفات التدريب الأصلية.
    - يحفظ النتائج في ملفات جديدة داخل MODEL_DIR.
    """

    # 1. إنشاء Generators (يشمل الصور الجديدة إذا تمت إضافتها)
    train_generator, val_generator, test_generator = create_data_generators()

    # 2. تحديد مسارات الملفات
    base_model_path = os.path.join(MODEL_DIR, start_checkpoint)
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)
    log_path = os.path.join(MODEL_DIR, 'trainig_log.csv')
    #log_path_new = os.path.join(MODEL_DIR, 'fine_tune_log.csv')
    fine_tuned_model_path = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')

    # 3. تحميل النموذج
    if resume and os.path.exists(checkpoint_path):
        print("\nresume training from the last point....")
        model = tf.keras.models.load_model(checkpoint_path)
        initial_epoch = 0
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            initial_epoch = len(log_df)
            print(f"resume training from epoch {initial_epoch}")
    else:
        print("\ndownload the basic model....")
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"the file {base_model_path}  is not found")
        model = tf.keras.models.load_model(base_model_path)
        initial_epoch = 0
        print("start new Fine-Tuning ...")

    # 4. إعادة تهيئة الـ optimizer بمعدل تعلم جديد
    model.compile(
        optimizer=Adam(learning_rate=new_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. إعداد الـ Callbacks (للحفظ والـ logging)
    callbacks = [
        ModelCheckpoint(fine_tuned_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ModelCheckpoint(checkpoint_path, save_freq="epoch", save_weights_only=False),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
        CSVLogger(log_path, append=True)
    ]
    target_epochs =initial_epoch + new_epochs 
    # 6. بدء Fine-Tuning (مع استئناف الإبوك إن وجد)
    print(f"\n start fine-Tuning with learning rate {new_lr}، from Epoch {initial_epoch} to {new_epochs}  additional epoch..")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // Patch_size,
        epochs=target_epochs,  # يكمل من حيث توقف
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        validation_steps=val_generator.samples // Patch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 7. حفظ النموذج النهائي بعد Fine-Tuning
    model.save(os.path.join(MODEL_DIR, 'final_fine_tuned_model.keras'))
    print("\n fine tunning has been completed and the model is saved")

    # 8. رسم منحنيات التدريب والـ validation
    plot_training_history(history)

    return model, history, test_generator


if __name__ == '__main__':
    # تشغيل Fine-Tuning (يمكن تعديل القيم حسب الحاجة)
    fine_tune_model(
        start_checkpoint='best_model.keras',   # النموذج الأساسي
        checkpoint_name='trainig_checkpoint.keras',  # نفس نقطة الاستئناف
        new_epochs=10,       # عدد الإيبوكس الإضافية
        new_lr=1e-5,         # معدل تعلم منخفض للضبط الدقيق
        resume=True          # استئناف التدريب إن وجد checkpoint
    )
