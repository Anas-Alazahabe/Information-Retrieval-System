import ir_datasets
import mysql.connector

# 1. إعدادات الاتصال (تمت إضافة utf8mb4 ليدعم كل الرموز)
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "MySecretPassword", 
    "database": "ir_system",
    "charset": "utf8mb4" 
}

def migrate_data():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        print("Creating table...")
        # استخدام utf8mb4 للجدول ليطابق الإعدادات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
        """)
        
        print("Loading dataset...")
        dataset = ir_datasets.load("msmarco-passage")
        
        batch_size = 1000
        batch = []
        count = 0
        
        print("Starting data insertion...")
        sql = "INSERT IGNORE INTO documents (id, content) VALUES (%s, %s)"
        
        for doc in dataset.docs_iter():
            batch.append((doc.doc_id, doc.text))
            count += 1
            
            if len(batch) == batch_size:
                try:
                    # محاولة إدخال الـ 1000 دفعة واحدة
                    cursor.executemany(sql, batch)
                    conn.commit()
                except mysql.connector.Error:
                    # في حال فشل الدفعة بسبب نص غريب، نتراجع ونحاول إدخالهم واحداً تلو الآخر
                    conn.rollback()
                    for item in batch:
                        try:
                            cursor.execute(sql, item)
                            conn.commit()
                        except:
                            conn.rollback() # تجاهل المستند المعطوب المسبب للمشكلة
                            
                print(f"Processed {count} records...")
                batch = [] 
                
            if count >= 200000:
                break
                
        # إدخال ما تبقى
        if batch:
            try:
                cursor.executemany(sql, batch)
                conn.commit()
            except:
                pass
            
        print("✅ تم نقل البيانات إلى قاعدة البيانات بنجاح!")
        
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
    except Exception as e:
        print(f"System Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    migrate_data()