from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

font_prop = FontProperties(fname='./AaXinRui85-2.ttf')
plt.figure(figsize=(6, 3))
plt.text(0.5, 0.5, '测试字体：你好，世界！AaXinRui85-2.ttf', fontproperties=font_prop, fontsize=24, ha='center', va='center')
plt.axis('off')
plt.savefig('font_test.png')
plt.show() 