
###########################################
# İş Problemi
###########################################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik,
# tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis
# ve kategorileri içeren veri setini kullanarak Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


###########################################
# Veri Seti Hikayesi
###########################################

# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir.
# (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


###########################################
# 1. Veriyi Hazırlama
###########################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv('armut_data.csv')
df = df_.copy()
df.head()

df.shape
df.info()
df.describe().T
df.isnull().sum()

df["Hizmet"] = df["ServiceId"].astype(str) + '_' + df["CategoryId"].astype(str)
df["New_Date"] = pd.to_datetime(df["CreateDate"]).dt.strftime('%Y-%m')
df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"]

###########################################
# 2. Birliktelik Kuralları Üretme
###########################################

df.pivot_table(index=["SepetID"],
               columns=["Hizmet"],
               values="")

invoice_product_df = df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

frequent_itemsets = apriori(invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
rules.head()

###########################################
# 3. Öneride Bulunma
###########################################

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values(by="lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


arl_recommender(rules, "2_0", 4)








