def show_box(img,predictions):
    extra_fields = predictions.extra_fields
    masks = extra_fields['mask']
    labels = extra_fields['labels']
    for mask in masks:
        mask = torch.squeeze(mask)
        segmentation = binary_mask_to_polygon(mask, tolerance=2)[0]
        seg_list.append(segmentation)
def show_mask(img, predictions, coco_demo):
    """
    显示分割效果
    :param img: numpy的图像 
    :param predictions: 分割结果
    :param coco_demo: 函数集
    :return: 显示分割效果
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 图片尺寸
    plt.imshow(img)  # 需要提前填充图像
    # plt.show()

    extra_fields = predictions.extra_fields
    masks = extra_fields['mask']
    labels = extra_fields['labels']
    name_list = [coco_demo.CATEGORIES[l] for l in labels]

    seg_list = []
    for mask in masks:
        mask = torch.squeeze(mask)
        segmentation = binary_mask_to_polygon(mask, tolerance=2)[0]
        seg_list.append(segmentation)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    polygons, color = [], []

    np.random.seed(37)

    for name, seg in zip(name_list, seg_list):
        c = (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]

        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        c_x, c_y = get_center_of_polygon(poly)  # 计算多边形的中心点

        # 0~26是大类别, 其余是小类别 同时 每个标签只绘制一次
        tc = c - np.array([0.5, 0.5, 0.5])  # 降低颜色
        tc = np.maximum(tc, 0.0)  # 最小值0

        plt.text(c_x, c_y, name, ha='left', wrap=True, color=tc,
                 bbox=dict(facecolor='white', alpha=0.5))  # 绘制标签

        polygons.append(pylab.Polygon(poly))  # 绘制多边形
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)  # 添加多边形
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)  # 添加多边形的框
    ax.add_collection(p)

    plt.axis('off')
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis

    out_folder = os.path.join(ROOT_DIR, 'demo', 'out')
    mkdir_if_not_exist(out_folder)
    out_file = os.path.join(out_folder, 'test.png'.format())
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()  # 避免所有图像绘制在一起

作者：SpikeKing
链接：https://juejin.im/post/5cde298af265da7e603903b0
来源：掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。