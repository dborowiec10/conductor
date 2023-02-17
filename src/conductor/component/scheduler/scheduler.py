

from conductor.component.scheduler.template import TemplateScheduler
from conductor.component.scheduler.sketch import SketchScheduler
from conductor.component.scheduler.flex import FlexScheduler

schedulers = {
    "template": TemplateScheduler,
    "sketch": SketchScheduler,
    "flex": FlexScheduler
}