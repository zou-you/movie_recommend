import gradio as gr
from movie_recommend import Recommend


recommend = Recommend()


with gr.Blocks(title='影视推荐系统') as demo:
    with gr.Row():
        name = gr.Textbox(label="请输入电影名", max_lines=1000)
    
    #设置按钮
    greet_btn = gr.Button("搜索")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            output1 = gr.Dataframe(headers=['电影名称', '评分', '类型', '导演', '主演', '剧情简介'], label='输入的电影信息', wrap=True)
            output2 = gr.Dataframe(headers=['序号', '电影名称', '评分', '类型', '导演', '主演', '剧情简介'], label='推荐电影列表', wrap=True)

    #设置按钮点击事件
    greet_btn.click(fn=recommend.recommend, inputs=name, outputs=[output1, output2])

demo.launch(server_name='0.0.0.0', share=True)