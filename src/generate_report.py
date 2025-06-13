from jinja2 import Environment, FileSystemLoader
import pandas as pd

def render_report(context, template_path='../Reports', template_name='Template.md', output_file='../Reports/EDA_Report.md'):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_name)
    report_md = template.render(context)

    with open(output_file, 'w') as f:
        f.write(report_md)
    print(f"✅ Report saved to {output_file}")