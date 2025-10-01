import streamlit as st, cv2, numpy as np, onnxruntime as ort, tempfile, os
from skimage.morphology import skeletonize

@st.cache_resource
def load_sess(onnx_path):
    return ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

def infer(sess, img, size=512):
    imr = cv2.resize(img, (size,size))
    x = imr[:,:,::-1].astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))[None]
    y = sess.run(None, {'input': x})[0][0,0]
    prob = (y*255).astype(np.uint8)
    mask = (prob>=128).astype(np.uint8)
    skel = skeletonize(mask>0).astype(np.uint8)*255
    return imr, prob, mask*255, skel

st.set_page_config(page_title="TinyCrack", layout="wide")
st.title("TinyCrack – Crack Detection & Aging Assessment")

onnx_path = st.sidebar.text_input("ONNX model path", "outputs/onnx/tinycrack.onnx")
sess = load_sess(onnx_path)

c1,c2 = st.columns(2)
with c1: up1 = st.file_uploader("Upload Date 1 image", type=['png','jpg','jpeg'])
with c2: up2 = st.file_uploader("Upload Date 2 image (optional)", type=['png','jpg','jpeg'])

if up1:
    img1 = cv2.imdecode(np.frombuffer(up1.read(), np.uint8), cv2.IMREAD_COLOR)
    imr1, prob1, mask1, sk1 = infer(sess, img1)
    st.subheader("Date 1")
    st.image([imr1[:,:,::-1], cv2.addWeighted(imr1,0.7, cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR),0.3,0)], caption=["Original","Overlay"], width=480)

    area1 = float((mask1>0).sum())/mask1.size
    len1 = int((sk1>0).sum())
    metrics = {"area_frac": round(area1,6), "length_px": len1}

    if up2:
        img2 = cv2.imdecode(np.frombuffer(up2.read(), np.uint8), cv2.IMREAD_COLOR)
        # Register date2->date1 (quick ORB+H)
        orb = cv2.ORB_create(2000)
        k1,d1 = orb.detectAndCompute(imr1, None)
        k2,d2 = orb.detectAndCompute(cv2.resize(img2,(imr1.shape[1],imr1.shape[0])), None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(d1,d2), key=lambda x:x.distance)[:500]
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
        H,_ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
        reg2 = cv2.warpPerspective(img2, H, (imr1.shape[1], imr1.shape[0]))

        imr2, prob2, mask2, sk2 = infer(sess, reg2)
        st.subheader("Date 2 (registered)")
        st.image([imr2[:,:,::-1], cv2.addWeighted(imr2,0.7, cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR),0.3,0)], caption=["Original","Overlay"], width=480)

        area2 = float((mask2>0).sum())/mask2.size
        len2 = int((sk2>0).sum())

        st.markdown("### Metrics")
        st.dataframe({
            "metric":["area_frac","length_px"],
            "date1":[area1,len1],
            "date2":[area2,len2],
            "delta":[area2-area1, len2-len1]
        })

        # CSV export
        if st.button("Export CSV"):
            import pandas as pd, io
            df = pd.DataFrame({"metric":["area_frac","length_px"],
                               "date1":[area1,len1],
                               "date2":[area2,len2],
                               "delta":[area2-area1, len2-len1]})
            st.download_button("Download metrics.csv", df.to_csv(index=False), "metrics.csv", "text/csv")

    else:
        st.json(metrics)
